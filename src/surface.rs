//! The panel's frame buffer, and the step that turns a recorded frame into panel pixels.
//!
//! [`Renderer`] records; this is where the recording becomes pixels. A frame passes through two
//! buffers and nothing else: [`Scene`], which remembers the commands the last frame was made of so
//! that this one can be diffed against them, and the RGB565 frame buffer the panel is written
//! from. There is no 8888 intermediate and no per-frame conversion — that is the whole point of
//! drawing through `pomelo-gfx` rather than through `tiny-skia`.
//!
//! The buffer is *ours* because it is the display's: the platform layer hands the pixels to the
//! panel and never looks inside them. `iced_tiny_skia` draws the same line between its compositor
//! (which owns a window's buffer) and the shell (which owns the window).

use iced_core::{Color, Rectangle, Size};
use iced_graphics::Viewport;
use pomelo_gfx::Pixmap565;

use crate::scene::Scene;
use crate::Renderer;

/// The panel's pixels, and what was drawn into them last frame.
pub struct Surface {
    panel: Pixmap565,
    recorded: Scene,
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
            recorded: Scene::new(),
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

    /// Replays a frame's recording into the panel buffer, and returns the rectangles that changed,
    /// in physical pixels.
    ///
    /// Unlike the `tiny-skia` path, which has to union its damage into a single rectangle —
    /// `tiny-skia` rasterises a primitive over its full extent and rejects pixels at blend time, so
    /// every extra rectangle is another full pass — this replays the frame's commands against each
    /// damage rectangle as it is, because `pomelo-gfx` takes the clip into the scan. What falls
    /// outside the damage costs nothing at all.
    ///
    /// An empty result means nothing moved and the panel does not need to be touched at all.
    pub fn present(&mut self, renderer: &Renderer, background: Color) -> Vec<Rectangle> {
        let screen = Rectangle::with_size(self.viewport.logical_size());
        let changed = self.recorded.advance(renderer.items());

        // A changed background invalidates every pixel. It is also not a command the tree drew —
        // the background belongs to the window, not to the tree — so it is painted here, under the
        // frame's own commands, which is what `iced_tiny_skia`'s `draw(…, background)` does in one
        // call of its own.
        let repaint = self.background != background;
        self.background = background;

        let damage = if repaint { vec![screen] } else { changed };

        let damage = iced_graphics::damage::group(damage, screen);

        if damage.is_empty() {
            return Vec::new();
        }

        let damage = iced_graphics::damage::group(damage, screen);

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
            // moved damages where it *was* as well as where it went, and nothing in the recording
            // draws there any more: without this, the square that left would still be on the
            // panel. `iced_tiny_skia`'s `draw(…, background)` does the same thing in one call.
            canvas.save();
            canvas.clip_rect(rect);
            canvas.clear(crate::geometry::color_of(background));

            renderer.replay(&mut canvas, rect);

            canvas.restore();
        }

        damage
            .iter()
            .map(|bounds| *bounds * self.viewport.scale_factor())
            .collect()
    }
}
