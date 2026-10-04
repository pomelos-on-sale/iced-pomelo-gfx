//! The panel's frame buffer, and the step that turns a recorded frame into panel pixels.
//!
//! [`Renderer`] records; this is where the recording becomes pixels. A frame passes through two
//! buffers and nothing else: the layers the last frame was made of — kept here so that this one can
//! be diffed against them, the way iced's own compositor keeps its layer stack — and the RGB565
//! frame buffer the panel is written from. There is no 8888 intermediate and no per-frame
//! conversion — that is the whole point of drawing through `pomelo-gfx` rather than through
//! `tiny-skia`.
//!
//! The buffer is *ours* because it is the display's: the platform layer hands the pixels to the
//! panel and never looks inside them. `iced_tiny_skia` draws the same line between its compositor
//! (which owns a window's buffer) and the shell (which owns the window).

use iced_core::{Color, Rectangle, Size};
use iced_graphics::Viewport;
use pomelo_gfx::Pixmap565;

use crate::layer::Layer;
use crate::Renderer;

/// The panel's pixels, and the frame they were last drawn from.
pub struct Surface {
    panel: Pixmap565,
    /// The layers the last frame was made of, which is what this one is compared against.
    ///
    /// `None` until the first frame has been drawn, which is what makes that one damage the whole
    /// panel: iced's own compositor gets the same effect from its buffer age.
    last: Option<Vec<Layer>>,
    /// The background the panel was last cleared to, because a change to it invalidates every
    /// pixel: it is painted under the frame's own commands, so nothing in the recording can
    /// express it.
    background: Color,
    viewport: Viewport,
}

impl Surface {
    /// Allocates the buffers for a panel of `width * height` physical pixels.
    ///
    /// `None` if the size is zero or the allocation fails — the panel is 480×480, so the frame
    /// buffer is 450 KiB and must land in PSRAM rather than internal SRAM.
    pub fn new(width: u32, height: u32) -> Option<Self> {
        Some(Self {
            panel: Pixmap565::new(width, height)?,
            last: None,
            // Deliberately not black: `present` treats a background change as "everything may
            // have moved", so this makes the first frame report full damage. iced's own
            // compositor gets the same effect by starting from `Color::TRANSPARENT`.
            background: Color::TRANSPARENT,
            // The panel is 1:1 — no HiDPI scaling anywhere in this stack.
            viewport: Viewport::with_physical_size(Size::new(width, height), 1.0),
        })
    }

    /// The viewport this surface draws for.
    pub fn viewport(&self) -> Viewport {
        self.viewport.clone()
    }

    /// The RGB565 frame buffer, ready to be handed to the panel.
    pub fn panel(&self) -> &Pixmap565 {
        &self.panel
    }

    /// Draws a frame's recording into the panel buffer, and returns the rectangles that changed,
    /// in physical pixels.
    ///
    /// Unlike the `tiny-skia` path, which has to union its damage into a single rectangle —
    /// `tiny-skia` rasterises a primitive over its full extent and rejects pixels at blend time, so
    /// every extra rectangle is another full pass — this draws the frame's commands against each
    /// damage rectangle as it is, because `pomelo-gfx` takes the clip into the scan. What falls
    /// outside the damage costs nothing at all.
    ///
    /// An empty result means nothing moved and the panel does not need to be touched at all.
    pub fn present(&mut self, renderer: &mut Renderer, background: Color) -> Vec<Rectangle> {
        let screen = Rectangle::with_size(self.viewport.logical_size());

        // A changed background invalidates every pixel. It is also not a command the tree drew —
        // the background belongs to the window, not to the tree — so it is painted here, under the
        // frame's own commands, which is what `iced_tiny_skia`'s `draw(…, background)` does in one
        // call of its own.
        let repaint = self.background != background;
        self.background = background;

        let damage = {
            let _damage = crate::profile::start(crate::profile::Phase::Damage);
            let current = renderer.layers();

            // The one call that makes this the same compositor as `iced_tiny_skia`'s: the same
            // helper, the same layer bounds, the same `Layer::damage` — and `Layer::damage` is a
            // refinement of iced's, so nothing is reported here that is not reported there.
            let damage = match &self.last {
                Some(previous) if !repaint => iced_graphics::damage::diff(
                    previous,
                    current,
                    |layer| vec![layer.bounds],
                    Layer::damage,
                ),
                _ => vec![screen],
            };

            self.last = Some(current.to_vec());

            let grouped = iced_graphics::damage::group(damage, screen);
            let aligned: Vec<Rectangle> = grouped
                .into_iter()
                .filter_map(|r| align_hardware_damage(r, screen))
                .collect();

            tighten(aligned)
        };

        // The frame is drawn first and reported after: the counters are the frame's, and this is
        // the only place that knows the frame has ended. A frame that draws nothing is reported too
        // — the phases it took are still a frame's — and a panel write is not in here at all,
        // because that belongs to the host and the host's own log has it.
        let damaged = if damage.is_empty() {
            Vec::new()
        } else {
            let mut canvas = pomelo_gfx::Canvas::new(self.panel.as_mut());

            for bounds in &damage {
                let bounds = *bounds * self.viewport.scale_factor();
                let rect = pomelo_gfx::Rect::from_ltrb(
                    bounds.x,
                    bounds.y,
                    bounds.x + bounds.width,
                    bounds.y + bounds.height,
                );

                // The background belongs to the window and not to the tree, so clearing the damage
                // back to it is this path's own job — and it is not an optimisation. A widget that
                // moved damages where it *was* as well as where it went, and nothing in the
                // recording draws there any more: without this, the square that left would still be
                // on the panel. `iced_tiny_skia`'s `draw(…, background)` does the same thing in one
                // call.
                let clear = crate::profile::start(crate::profile::Phase::Clear);

                canvas.save();
                canvas.clip_rect(rect);
                canvas.clear(crate::geometry::color_of(background));

                drop(clear);

                renderer.draw(&mut canvas, rect);

                canvas.restore();
            }

            damage
                .iter()
                .map(|bounds| *bounds * self.viewport.scale_factor())
                .collect()
        };

        crate::profile::report();

        damaged
    }
}

/// Snaps a damage rectangle to 2-pixel aligned integer boundaries.
///
/// The CO5300 AMOLED QSPI controller hardware constraint requires coordinates to be 2-pixel (even)
/// aligned. Snapping `(x1, y1)` down to the nearest even integer and `(x2, y2)` up to the nearest
/// even integer guarantees that `canvas.clear`, `renderer.draw` and the display DMA write path
/// (`hal_display_draw_bitmap`) operate on the exact same pixels, preventing un-cleared border pixels
/// or DMA 1-pixel shifts from displaying noisy vertical lines on region edges.
fn align_hardware_damage(rect: Rectangle, screen: Rectangle) -> Option<Rectangle> {
    let x1 = ((rect.x.floor() as i32).max(0) / 2 * 2) as f32;
    let y1 = ((rect.y.floor() as i32).max(0) / 2 * 2) as f32;
    let x2 = (((rect.x + rect.width).ceil() as i32 + 1) / 2 * 2).min(screen.width as i32) as f32;
    let y2 = (((rect.y + rect.height).ceil() as i32 + 1) / 2 * 2).min(screen.height as i32) as f32;

    if x2 > x1 && y2 > y1 {
        Some(Rectangle {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        })
    } else {
        None
    }
}

/// Folds a damage list into itself wherever doing so cannot cost more pixels.
///
/// iced's [`iced_graphics::damage::group`] merges in a single pass with one accumulator, sorted by
/// distance from the origin, so a region that overlaps one it has *already emitted* is never
/// revisited: a whole-screen frame can come back as the screen **plus rectangles inside it**.
/// `iced_tiny_skia` gets away with that because it rasterises every primitive over its full extent
/// per rectangle — but this path draws each rectangle on its own, so a rectangle that is already
/// inside another one is a second pass over the same pixels. Measured on the launcher opening
/// Settings: 7 rectangles and 344,743 px of drawing for a 230,400 px panel.
///
/// Two rectangles are folded when their union costs no more than the two of them together — always
/// true when they overlap, never true when they are far apart. **The result never paints more
/// pixels than the list it was given**, and it never has more rectangles. Disjoint damage stays
/// separate, which is the thing this path exists for and `tiny-skia` cannot afford.
fn tighten(mut damage: Vec<Rectangle>) -> Vec<Rectangle> {
    let mut folded = true;

    while folded {
        folded = false;

        let mut output: Vec<Rectangle> = Vec::with_capacity(damage.len());

        'rectangles: for bounds in damage {
            for existing in output.iter_mut() {
                let union = existing.union(&bounds);

                if union.area() <= existing.area() + bounds.area() {
                    *existing = union;
                    folded = true;
                    continue 'rectangles;
                }
            }

            output.push(bounds);
        }

        damage = output;
    }

    damage
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_core::Point;

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rectangle {
        Rectangle::new(Point::new(x, y), Size::new(width, height))
    }

    /// The measured case: the screen, plus three of the rectangles iced's grouping emits inside it
    /// (a card, a row and a label). Painting the screen already paints them.
    #[test]
    fn a_rectangle_inside_another_one_is_folded_into_it() {
        let screen = rect(0.0, 0.0, 480.0, 480.0);

        assert_eq!(
            tighten(vec![
                rect(19.0, 59.0, 442.0, 51.0),
                screen,
                rect(73.0, 220.0, 49.0, 21.0),
                rect(0.0, 0.0, 272.0, 246.0),
            ]),
            vec![screen]
        );
    }

    /// Two overlapping rectangles are one pass over the frame's commands instead of two, whenever
    /// the union is not more pixels than the pair.
    #[test]
    fn overlapping_rectangles_become_one() {
        assert_eq!(
            tighten(vec![
                rect(0.0, 0.0, 100.0, 100.0),
                rect(50.0, 0.0, 100.0, 100.0),
            ]),
            vec![rect(0.0, 0.0, 150.0, 100.0)]
        );
    }

    /// Overlapping *corners* are left alone: the union would cover 22,500 px for two rectangles of
    /// 10,000, which is more work, not less.
    #[test]
    fn rectangles_that_only_touch_a_corner_stay_separate() {
        let damage = vec![rect(0.0, 0.0, 100.0, 100.0), rect(50.0, 50.0, 100.0, 100.0)];

        assert_eq!(tighten(damage.clone()), damage);
    }

    /// Disjoint damage stays disjoint: the whole point of drawing per rectangle is that far-apart
    /// changes do not pay for the space between them.
    #[test]
    fn rectangles_far_apart_stay_separate() {
        let damage = vec![rect(0.0, 0.0, 10.0, 10.0), rect(400.0, 400.0, 10.0, 10.0)];

        assert_eq!(tighten(damage.clone()), damage);
    }

    /// Odd and subpixel boundaries are snapped outward to even integer coordinates for hardware QSPI alignment.
    #[test]
    fn damage_is_aligned_to_even_hardware_grid() {
        let screen = rect(0.0, 0.0, 480.0, 480.0);

        // Odd x (73.0), fractional width (49.3): x1=72, x2=ceil(122.3)=123 -> 124, width=52
        // Odd y (59.7), fractional height (21.1): y1=58, y2=ceil(80.8)=81 -> 82, height=24
        let unaligned = rect(73.0, 59.7, 49.3, 21.1);
        let aligned = align_hardware_damage(unaligned, screen).expect("aligned rect exists");

        assert_eq!(aligned.x, 72.0);
        assert_eq!(aligned.y, 58.0);
        assert_eq!(aligned.width, 52.0);
        assert_eq!(aligned.height, 24.0);
        assert_eq!(aligned.x % 2.0, 0.0);
        assert_eq!(aligned.y % 2.0, 0.0);
        assert_eq!((aligned.x + aligned.width) % 2.0, 0.0);
        assert_eq!((aligned.y + aligned.height) % 2.0, 0.0);
    }
}
