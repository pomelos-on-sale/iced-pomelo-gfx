//! The damage between two frames of **our own** recording.
//!
//! The platform layer's `damage.rs` does the same job for `iced_tiny_skia`'s layers, and needs
//! three tricks to do it: iced's own `Layer::damage` pairs items by index (so an insertion in the
//! middle of a list over-reports), text is compared through a closure that skips its content, and
//! the quads have to be pulled out of the layers entirely because they are painted in z-order and
//! paired by geometry instead. None of that is needed here, and the reason is the recording:
//! [`Item`] is one flat list of commands, each already carrying its clip, so the damage is a
//! merge-join over that list and nothing else.
//!
//! What it buys, in one line: a canvas that redraws a stroke animation appends the strokes that are
//! already finished as *equal* items, so the frame damages the stroke that is still growing rather
//! than the whole picture. `iced_tiny_skia` cannot see that — one canvas is one indivisible item
//! there — which is why hello's animation is the test case for this renderer.
//!
//! # Why geometry, and not the index
//!
//! The two lists are sorted by bounds first and then walked together. A merge-join visits items in
//! the same order whenever their geometry is the same, so an item that appears in the middle of a
//! frame costs that item and nothing after it — which is exactly what index pairing gets wrong, and
//! what was measured on the launcher: one tile's pressed wash is one more quad, and pairing by
//! index damaged 82,027 px where the tile's own wash is 19,733.
//!
//! Sorting is stable, which is what makes two commands in the *same* rectangle (a fill and its
//! border) keep their relative order and pair with their own kind rather than swapping.

use std::cmp::Ordering;

use iced_core::Rectangle;

use crate::Item;

/// What was drawn last frame, in the form the next frame's damage is computed against.
pub struct Scene {
    items: Vec<Item>,
}

impl Scene {
    /// An empty scene, which damages everything it is first compared against.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// The damage between this scene and `current`, and this scene replaced by it.
    pub fn advance(&mut self, current: &[Item]) -> Vec<Rectangle> {
        let damage = damage(&self.items, current);

        self.items = current.to_vec();

        damage
    }

    /// What was drawn last frame.
    pub fn items(&self) -> &[Item] {
        &self.items
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

/// The damage between two frames' commands.
///
/// Both lists are sorted by bounds, and then merged. The cases are the four that exist: a command
/// only the old frame had, one only the new frame has, one whose rectangle is the same but whose
/// contents changed, and one that did not move at all — which is the case worth having.
///
/// A rectangle can be reported more than once, and that is not a defect being tolerated: two
/// commands in the same rectangle (a fill and its border) are paired in list order, so if they swap
/// places both positions find *something* changed. Reporting the same rectangle twice costs a
/// comparison and not a pixel — the caller groups the rectangles before painting — but it is why
/// the damage is a list and not a set.
fn damage(previous: &[Item], current: &[Item]) -> Vec<Rectangle> {
    let mut before: Vec<&Item> = previous.iter().collect();
    let mut after: Vec<&Item> = current.iter().collect();

    before.sort_by(|a, b| by_bounds(&a.bounds(), &b.bounds()));
    after.sort_by(|a, b| by_bounds(&a.bounds(), &b.bounds()));

    let mut damage = Vec::new();
    let (mut i, mut j) = (0, 0);

    loop {
        match (before.get(i), after.get(j)) {
            (None, None) => break,
            (Some(previous), None) => {
                damage.extend(clipped(&previous.bounds(), previous.clip));
                i += 1;
            }
            (None, Some(now)) => {
                damage.extend(clipped(&now.bounds(), now.clip));
                j += 1;
            }
            (Some(previous), Some(now)) => {
                match by_bounds(&previous.bounds(), &now.bounds()) {
                    // The same rectangle in both frames: damage it only if what was drawn in it
                    // changed. This is the case the whole design exists for.
                    Ordering::Equal => {
                        if previous.primitive != now.primitive {
                            damage.extend(clipped(&now.bounds(), now.clip));
                        }

                        i += 1;
                        j += 1;
                    }
                    // Otherwise one of them has no counterpart in the other frame, and its own
                    // rectangle is what changed.
                    Ordering::Less => {
                        damage.extend(clipped(&previous.bounds(), previous.clip));
                        i += 1;
                    }
                    Ordering::Greater => {
                        damage.extend(clipped(&now.bounds(), now.clip));
                        j += 1;
                    }
                }
            }
        }
    }

    damage
}

/// The rectangle a quad's damage occupies: grown by the pixel its antialiased edge bleeds into,
/// and clipped to the layer it was drawn in. The same rectangle iced reports for one quad.
///
/// The platform layer's `damage.rs` has its own pair of these — the two scenes live in different
/// crates now, and two eight-line functions shared between them would mean sharing the whole
/// crate, which is the wrong reason to.
fn clipped(bounds: &Rectangle, clip: Rectangle) -> Option<Rectangle> {
    bounds.expand(1.0).intersection(&clip)
}

/// The order that makes the pairing geometric, and that both scenes sort by.
fn by_bounds(a: &Rectangle, b: &Rectangle) -> Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.width.total_cmp(&b.width))
        .then_with(|| a.height.total_cmp(&b.height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_core::{Color, Point, Size};

    use crate::geometry::{self, Primitive};
    use crate::Placement;
    use pomelo_gfx::Rect as GfxRect;

    const SCREEN: Rectangle = Rectangle {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
    };

    /// A command that fills an 8x8 square at `(x, 0)`.
    fn square(x: f32) -> Item {
        Item {
            primitive: Primitive::Rect {
                rect: GfxRect::from_ltwh(x, 0.0, 8.0, 8.0),
                paint: geometry::solid_paint(Color::WHITE),
            },
            clip: SCREEN,
            placement: Placement::IDENTITY,
        }
    }

    /// The same command, filled in another colour.
    fn square_in(x: f32, color: Color) -> Item {
        Item {
            primitive: Primitive::Rect {
                rect: GfxRect::from_ltwh(x, 0.0, 8.0, 8.0),
                paint: geometry::solid_paint(color),
            },
            clip: SCREEN,
            placement: Placement::IDENTITY,
        }
    }

    /// The rectangle a square's damage occupies: its own bounds, grown by the pixel its edge
    /// bleeds into, and clipped to the screen -- the definition the renderer's damage follows.
    fn covered(x: f32) -> Rectangle {
        Rectangle::new(Point::new(x - 1.0, -1.0), Size::new(10.0, 10.0))
            .intersection(&SCREEN)
            .expect("a square on the screen")
    }

    #[test]
    fn the_first_frame_damages_everything_recorded() {
        let mut scene = Scene::new();

        let damage = scene.advance(&[square(0.0), square(32.0)]);

        assert_eq!(damage, vec![covered(0.0), covered(32.0)]);
    }

    #[test]
    fn an_identical_frame_damages_nothing() {
        let mut scene = Scene::new();
        let frame = [square(0.0), square(32.0)];

        scene.advance(&frame);

        assert!(
            scene.advance(&frame).is_empty(),
            "a frame that drew the same commands has changed nothing"
        );
    }

    #[test]
    fn a_command_that_changed_damages_its_own_rectangle() {
        let mut scene = Scene::new();

        scene.advance(&[square(0.0), square(32.0)]);

        let damage = scene.advance(&[square_in(0.0, Color::WHITE), square_in(32.0, Color::BLACK)]);

        assert_eq!(
            damage,
            vec![covered(32.0)],
            "only the square whose contents changed"
        );
    }

    #[test]
    fn an_insertion_damages_the_insertion_and_nothing_after_it() {
        let mut scene = Scene::new();

        scene.advance(&[square(0.0), square(32.0), square(48.0)]);

        // One command appears in the middle of the list. Pairing by index would compare every item
        // after it against its neighbour and damage all of them; pairing by geometry stops at the
        // one that actually appeared.
        let damage = scene.advance(&[square(0.0), square(16.0), square(32.0), square(48.0)]);

        assert_eq!(damage, vec![covered(16.0)]);
    }

    #[test]
    fn a_command_that_went_away_damages_the_rectangle_it_left() {
        let mut scene = Scene::new();

        scene.advance(&[square(0.0), square(16.0), square(32.0)]);

        let damage = scene.advance(&[square(0.0), square(32.0)]);

        assert_eq!(damage, vec![covered(16.0)]);
    }

    #[test]
    fn two_commands_in_one_rectangle_damage_only_that_rectangle() {
        // A fill and its border are two commands with the same bounds. The merge-join pairs them
        // in list order, so when the two shuffle the rectangle is reported once per position --
        // two entries, the same rectangle. The caller groups them; what matters here is that no
        // *other* rectangle is reported, and that the area does not grow.
        let mut scene = Scene::new();

        scene.advance(&[square(0.0), square_in(0.0, Color::BLACK)]);

        let damage = scene.advance(&[square_in(0.0, Color::BLACK), square_in(0.0, Color::WHITE)]);

        assert!(!damage.is_empty(), "the rectangle has to be reported");
        assert!(
            damage.iter().all(|bounds| *bounds == covered(0.0)),
            "and nothing but that rectangle: {damage:?}"
        );
    }

    #[test]
    fn a_moved_command_damages_both_its_places() {
        let mut scene = Scene::new();

        scene.advance(&[square(0.0)]);

        let damage = scene.advance(&[square(32.0)]);

        assert_eq!(
            damage,
            vec![covered(0.0), covered(32.0)],
            "where it was and where it went"
        );
    }

    #[test]
    fn a_clipped_command_damages_only_what_the_clip_allowed() {
        let mut scene = Scene::new();

        let mut clipped = square(0.0);
        clipped.clip = Rectangle::new(Point::new(0.0, 0.0), Size::new(4.0, 64.0));

        let damage = scene.advance(&[clipped]);

        assert_eq!(
            damage,
            vec![Rectangle::new(Point::new(0.0, 0.0), Size::new(4.0, 9.0))],
            "the clip cuts the command's damage down to what it could paint"
        );
    }
}
