//! The layer a frame is recorded into, and the damage between two frames of it.
//!
//! The shape is `iced_tiny_skia`'s, method for method: a [`Layer`] holds the primitives that were
//! drawn under one clip, [`Item`] is how a canvas's own recording sits in it — the same
//! `Live`/`Group`/`Cached` three cases, with the cache compared by identity — and
//! [`Layer::damage`] answers what changed between two frames exactly where `Layer::damage` answers
//! it there. A [`Renderer`](crate::Renderer) records through iced's
//! [`layer::Stack`](iced_graphics::layer::Stack), so opening a clip and pushing a transformation
//! are the same two calls, and [`Surface`](crate::Surface) keeps the last frame's layers and diffs
//! them the way iced's own compositor does.
//!
//! Two things are deliberately not the same, and both are stated where they live:
//!
//! * **The payloads.** `quads` holds [`Primitive`]s — already converted for our rasteriser, and
//!   already placed, because the rasteriser takes a transform rather than a `Quad` — where
//!   `iced_tiny_skia` holds `(Quad, Background)` and converts while painting. The damage is the
//!   same either way: the conversion is total, and it keeps what the diff compares.
//! * **The pairing.** `iced_tiny_skia` pairs a layer's commands *by index*, so inserting one in
//!   the middle of a list damages everything after it. Here both frames' commands are sorted by
//!   the rectangle they occupy and merged, so an insertion costs the insertion and nothing else.
//!   That is a refinement, not a different answer: whenever index pairing reports nothing, so does
//!   this, and it reports a subset of the rectangles otherwise. The launcher is where it was
//!   measured: one tile's pressed wash is one more quad, 19,733 px this way against 82,027 the
//!   other.
//!
//! # Why a canvas is not one indivisible command here
//!
//! `iced_tiny_skia` records a canvas as one `Item::Group` and compares it all-or-nothing, so a
//! canvas that redraws a stroke animation damages its whole picture every frame — unless the app
//! puts the static part in a `canvas::Cache`, which is a pointer comparison and costs nothing.
//! Both of those work here as well, and on top of them a `Group` that *did* change is walked
//! command by command: the pieces that are byte-identical pair with themselves and only the piece
//! that is still growing damages anything. So the animation is cheap with a cache and cheap
//! without one.

use std::cmp::Ordering;
use std::sync::Arc;

use iced_core::{Color, Point, Rectangle, Size, Transformation};

use crate::geometry::{Primitive, TextRun};
use crate::renderer::Placement;

/// One recorded command: a command of its own, or a canvas's recording of many.
///
/// The three cases are `iced_tiny_skia`'s, and they mean what they mean there: a `Live` command is
/// compared by value, a `Cached` one by the identity of the buffer a `canvas::Cache` is holding —
/// so a cache that nobody cleared is unchanged without a single comparison — and a `Group` is a
/// canvas's live recording, whose clip and transformation travel with it.
#[derive(Debug, Clone, PartialEq)]
pub enum Item<T> {
    /// One command, drawn where it was recorded.
    Live(T),
    /// A canvas's recording, drawn through `clip_bounds` and `Transformation`.
    Group(Vec<T>, Rectangle, Transformation),
    /// A canvas's recording that a `canvas::Cache` is holding on to.
    Cached(Arc<[T]>, Rectangle, Transformation),
}

impl<T> Item<T> {
    /// The commands this item drew, in order.
    pub fn as_slice(&self) -> &[T] {
        match self {
            Item::Live(command) => std::slice::from_ref(command),
            Item::Group(commands, ..) => commands,
            Item::Cached(commands, ..) => commands,
        }
    }

    /// The rectangle a command was clipped to, in the recording's own coordinates.
    ///
    /// A `Live` command has no clip of its own: its layer's bounds are what it was drawn inside.
    pub fn clip_bounds(&self) -> Rectangle {
        match self {
            Item::Live(_) => Rectangle::INFINITE,
            Item::Group(_, clip_bounds, _) | Item::Cached(_, clip_bounds, _) => *clip_bounds,
        }
    }

    /// The transformation the item was recorded under.
    pub fn transformation(&self) -> Transformation {
        match self {
            Item::Live(_) => Transformation::IDENTITY,
            Item::Group(_, _, transformation) | Item::Cached(_, _, transformation) => {
                *transformation
            }
        }
    }
}

/// One text command: what it says, where, in what colour, and how far in.
///
/// `bounds` and `clip` are already in device space — the transformation in force when it was
/// recorded is baked into them, which is what `iced_tiny_skia` does too — and `transformation`
/// travels separately because the glyphs themselves are placed by it.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    /// Where the run's top-left goes, in the recording's own coordinates.
    pub position: Point,
    /// What colour the glyphs are drawn in.
    pub color: Color,
    /// The text's own box, in device space: all the damage a run can cause.
    pub bounds: Rectangle,
    /// What it was clipped to, in device space: its layer, narrowed by the widget.
    pub clip: Rectangle,
    /// What it says, in one of the two shapes iced hands text over in.
    pub run: TextRun,
    /// The transformation the run was recorded under, for placing the glyphs.
    pub transformation: Transformation,
}

/// A layer of graphical primitives: the commands drawn under one clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// The clip every command in the layer was drawn inside.
    pub bounds: Rectangle,
    /// The widget commands, already placed: fills, borders, shadows.
    pub quads: Vec<Primitive>,
    /// The canvas geometry, one item per canvas.
    pub primitives: Vec<Item<Primitive>>,
    /// The text, one item per run.
    pub text: Vec<Item<Text>>,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            bounds: Rectangle::INFINITE,
            quads: Vec::new(),
            primitives: Vec::new(),
            text: Vec::new(),
        }
    }
}

impl iced_graphics::layer::Layer for Layer {
    fn with_bounds(bounds: Rectangle) -> Self {
        Self {
            bounds,
            ..Self::default()
        }
    }

    fn bounds(&self) -> Rectangle {
        self.bounds
    }

    /// Nothing is pending: a command is recorded where it is drawn.
    fn flush(&mut self) {}

    fn resize(&mut self, bounds: Rectangle) {
        self.bounds = bounds;
    }

    fn reset(&mut self) {
        self.bounds = Rectangle::INFINITE;

        self.quads.clear();
        self.primitives.clear();
        self.text.clear();
    }

    /// Where the layer's primitive kinds start, for [`Stack`](iced_graphics::layer::Stack)'s
    /// merging.
    ///
    /// The levels are `iced_tiny_skia`'s, one per kind in painting order, and they are what lets
    /// two neighbouring layers be merged when their commands do not interleave.
    fn start(&self) -> usize {
        if !self.quads.is_empty() {
            return 1;
        }

        if !self.primitives.is_empty() {
            return 2;
        }

        3
    }

    /// Where the layer's primitive kinds end.
    fn end(&self) -> usize {
        if !self.text.is_empty() {
            return 3;
        }

        if !self.primitives.is_empty() {
            return 2;
        }

        1
    }

    fn merge(&mut self, layer: &mut Self) {
        self.quads.append(&mut layer.quads);
        self.primitives.append(&mut layer.primitives);
        self.text.append(&mut layer.text);
    }
}

impl Layer {
    /// The rectangles two frames disagree in.
    ///
    /// The layer's own bounds are compared first — a layer that moved is damage as a whole, which
    /// is what `iced_tiny_skia` does and what makes this cheap for the common case — and then each
    /// kind of command is walked. An empty answer means the two frames are the same picture and the
    /// panel does not have to be touched at all.
    pub fn damage(previous: &Self, current: &Self) -> Vec<Rectangle> {
        if previous.bounds != current.bounds {
            return vec![previous.bounds, current.bounds];
        }

        let mut damage = Vec::new();

        damage.extend(commands(
            &previous.quads,
            &current.quads,
            Placement::IDENTITY,
            current.bounds,
        ));
        damage.extend(items(
            &previous.primitives,
            &current.primitives,
            current.bounds,
        ));
        damage.extend(texts(&previous.text, &current.text, current.bounds));

        damage
    }
}

/// The damage between two frames' widget commands.
fn commands(
    previous: &[Primitive],
    current: &[Primitive],
    placement: Placement,
    clip: Rectangle,
) -> Vec<Rectangle> {
    walk(
        previous,
        current,
        |command| covering(&rects(command, placement, clip)),
        |command| rects(command, placement, clip),
        |previous, current| {
            if previous == current {
                Vec::new()
            } else {
                rects(current, placement, clip)
            }
        },
    )
}

/// The damage between two frames' text.
fn texts(previous: &[Item<Text>], current: &[Item<Text>], clip: Rectangle) -> Vec<Rectangle> {
    walk(
        previous,
        current,
        |item| covering(&text_rects(item, clip)),
        |item| text_rects(item, clip),
        |previous, current| {
            if previous == current {
                Vec::new()
            } else {
                text_rects(current, clip)
            }
        },
    )
}

/// The rectangles one text command occupies: its own box, inside what it was clipped to.
fn text_rects(item: &Item<Text>, clip: Rectangle) -> Vec<Rectangle> {
    item.as_slice()
        .iter()
        .filter_map(|text| {
            let clip = text.clip.intersection(&clip)?;

            clipped(text.bounds, clip)
        })
        .collect()
}

/// The damage between two frames' canvas geometry.
fn items(
    previous: &[Item<Primitive>],
    current: &[Item<Primitive>],
    clip: Rectangle,
) -> Vec<Rectangle> {
    walk(
        previous,
        current,
        |item| item_frame(item, clip),
        |item| item_rects(item, clip),
        |previous, current| item_damage(previous, current, clip),
    )
}

/// The rectangle a canvas item's frame occupies: its clip, placed, inside the layer.
///
/// This is what the pairing uses, and not its commands' own bounds: a canvas that is drawing an
/// animation has different commands every frame, while its frame is the same rectangle in both.
/// Pairing by the frame is what makes the walk inside it possible at all — and two canvases in the
/// same rectangle pair in the order they were drawn, like every other command here.
fn item_frame(item: &Item<Primitive>, clip: Rectangle) -> Rectangle {
    Placement::of(item.transformation())
        .map(item.clip_bounds())
        .intersection(&clip)
        .unwrap_or(Rectangle::with_size(Size::ZERO))
}

/// The rectangles a canvas item's commands occupy, in device space and inside the layer.
fn item_rects(item: &Item<Primitive>, clip: Rectangle) -> Vec<Rectangle> {
    let placement = Placement::of(item.transformation());

    let Some(clip) = placement.map(item.clip_bounds()).intersection(&clip) else {
        return Vec::new();
    };

    item.as_slice()
        .iter()
        .filter_map(|command| clipped(placement.map(command.bounds()), clip))
        .collect()
}

/// What two canvas items recorded into the same frame disagree in.
///
/// The caller only asks this of two items whose [`item_frame`] is the same rectangle, so this is
/// where a canvas is *not* one indivisible command: a cached item that nobody cleared is unchanged
/// by identity, and a group — or a cache that was rebuilt — is walked command by command, so the
/// pieces that did not move are not reported.
fn item_damage(
    previous: &Item<Primitive>,
    current: &Item<Primitive>,
    clip: Rectangle,
) -> Vec<Rectangle> {
    if let (Item::Cached(previous, ..), Item::Cached(current, ..)) = (previous, current) {
        if Arc::ptr_eq(previous, current) {
            return Vec::new();
        }
    }

    let placement = Placement::of(current.transformation());

    let Some(clip) = placement.map(current.clip_bounds()).intersection(&clip) else {
        return Vec::new();
    };

    commands(previous.as_slice(), current.as_slice(), placement, clip)
}

/// Walks two recordings of one kind together, by geometry.
///
/// Both are sorted by the rectangle that identifies a command — its own bounds for a leaf, its
/// frame for a canvas — and then merged. Four cases exist: a command only the old frame had, one
/// only the new frame has, one whose rectangle is the same but whose contents changed, and one that
/// did not move at all — which is the case worth having. Sorting is stable, so two commands with
/// the same rectangle (a fill and its border, two canvases over each other) keep their relative
/// order and pair with their own kind rather than swapping.
///
/// A rectangle can be reported more than once when two commands shuffle inside it. That costs a
/// comparison and not a pixel: the caller groups the rectangles before painting.
fn walk<T>(
    previous: &[T],
    current: &[T],
    key: impl Fn(&T) -> Rectangle,
    rects: impl Fn(&T) -> Vec<Rectangle>,
    changed: impl Fn(&T, &T) -> Vec<Rectangle>,
) -> Vec<Rectangle> {
    let before = sorted(previous, &key);
    let after = sorted(current, &key);

    let mut damage = Vec::new();
    let (mut i, mut j) = (0, 0);

    loop {
        match (before.get(i), after.get(j)) {
            (None, None) => break,
            (Some((previous, _)), None) => {
                damage.extend(rects(previous));
                i += 1;
            }
            (None, Some((current, _))) => {
                damage.extend(rects(current));
                j += 1;
            }
            (Some((previous, previous_key)), Some((current, current_key))) => {
                match by_bounds(previous_key, current_key) {
                    Ordering::Equal => {
                        damage.extend(changed(previous, current));
                        i += 1;
                        j += 1;
                    }
                    Ordering::Less => {
                        damage.extend(rects(previous));
                        i += 1;
                    }
                    Ordering::Greater => {
                        damage.extend(rects(current));
                        j += 1;
                    }
                }
            }
        }
    }

    damage
}

/// A frame's commands, each with the rectangle that identifies it.
fn sorted<'a, T>(commands: &'a [T], key: &impl Fn(&T) -> Rectangle) -> Vec<(&'a T, Rectangle)> {
    let mut sorted: Vec<(&T, Rectangle)> = commands
        .iter()
        .map(|command| (command, key(command)))
        .collect();

    sorted.sort_by(|(_, a), (_, b)| by_bounds(a, b));

    sorted
}

/// The smallest rectangle containing all of `rects`, and `Rectangle::with_size(Size::ZERO)` when
/// there are none.
fn covering(rects: &[Rectangle]) -> Rectangle {
    let mut rects = rects.iter();

    let Some(first) = rects.next() else {
        return Rectangle::with_size(Size::ZERO);
    };

    rects.fold(*first, |covering, rect| {
        let x = covering.x.min(rect.x);
        let y = covering.y.min(rect.y);

        let right = (covering.x + covering.width).max(rect.x + rect.width);
        let bottom = (covering.y + covering.height).max(rect.y + rect.height);

        Rectangle {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    })
}

/// The rectangles one command occupies: its own bounds, grown by the pixel its antialiased edge
/// bleeds into, and clipped to what it was allowed to paint.
fn rects(command: &Primitive, placement: Placement, clip: Rectangle) -> Vec<Rectangle> {
    clipped(placement.map(command.bounds()), clip)
        .into_iter()
        .collect()
}

/// A command's rectangle, grown by the pixel its antialiased edge bleeds into and clipped.
fn clipped(bounds: Rectangle, clip: Rectangle) -> Option<Rectangle> {
    bounds.expand(1.0).intersection(&clip)
}

/// The order that makes the pairing geometric: across, then down, then by size.
fn by_bounds(a: &Rectangle, b: &Rectangle) -> Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.width.total_cmp(&b.width))
        .then_with(|| a.height.total_cmp(&b.height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_core::text::{Alignment, Shaping};
    use iced_core::{Font, Size};
    use iced_graphics::layer::Layer as _;
    use pomelo_gfx::Rect as GfxRect;

    use crate::geometry::{self, Parameters};

    const SCREEN: Rectangle = Rectangle {
        x: 0.0,
        y: 0.0,
        width: 64.0,
        height: 64.0,
    };

    /// A layer of the screen holding `quads`.
    fn layer(quads: Vec<Primitive>) -> Layer {
        Layer {
            quads,
            ..Layer::with_bounds(SCREEN)
        }
    }

    /// A layer of the screen holding one canvas item.
    fn canvas(items: Vec<Item<Primitive>>) -> Layer {
        Layer {
            primitives: items,
            ..Layer::with_bounds(SCREEN)
        }
    }

    /// A command that fills an 8x8 square at `(x, 0)`.
    fn square(x: f32) -> Primitive {
        square_w(x, 8.0)
    }

    /// The same command, filled in another colour.
    fn square_in(x: f32, color: Color) -> Primitive {
        Primitive::Rect {
            rect: GfxRect::from_ltwh(x, 0.0, 8.0, 8.0),
            paint: geometry::solid_paint(color),
        }
    }

    /// A command that fills a `width` by 8 rectangle at `(x, 0)`: a tip that grows.
    fn square_w(x: f32, width: f32) -> Primitive {
        Primitive::Rect {
            rect: GfxRect::from_ltwh(x, 0.0, width, 8.0),
            paint: geometry::solid_paint(Color::WHITE),
        }
    }

    /// A canvas item recording `commands`.
    fn group(commands: Vec<Primitive>) -> Item<Primitive> {
        Item::Group(commands, SCREEN, Transformation::IDENTITY)
    }

    /// A canvas item a `canvas::Cache` is holding on to.
    fn cached(commands: Vec<Primitive>) -> Item<Primitive> {
        Item::Cached(Arc::from(commands), SCREEN, Transformation::IDENTITY)
    }

    /// A run of text saying `content`.
    fn text(content: &str) -> Item<Text> {
        Item::Live(Text {
            position: Point::ORIGIN,
            color: Color::WHITE,
            bounds: Rectangle::new(Point::ORIGIN, Size::new(20.0, 10.0)),
            clip: SCREEN,
            run: TextRun::Parameters(Parameters {
                content: String::from(content),
                size: 16.0,
                line_height: 16.0,
                font: Font::default(),
                align_x: Alignment::Left,
                shaping: Shaping::Basic,
                bounds: Size::new(20.0, 10.0),
            }),
            transformation: Transformation::IDENTITY,
        })
    }

    /// The rectangle a `width` by 8 command's damage occupies: its own bounds, grown by the pixel
    /// its edge bleeds into, and clipped to the screen.
    fn covered_w(x: f32, width: f32) -> Rectangle {
        Rectangle::new(Point::new(x - 1.0, -1.0), Size::new(width + 2.0, 10.0))
            .intersection(&SCREEN)
            .expect("a command on the screen")
    }

    /// The same, for the 8x8 squares the tests are written with.
    fn covered(x: f32) -> Rectangle {
        covered_w(x, 8.0)
    }

    #[test]
    fn a_layer_that_drew_nothing_is_damaged_by_everything_drawn() {
        let damage = Layer::damage(&layer(Vec::new()), &layer(vec![square(0.0), square(32.0)]));

        assert_eq!(damage, vec![covered(0.0), covered(32.0)]);
    }

    #[test]
    fn an_identical_frame_damages_nothing() {
        let frame = || layer(vec![square(0.0), square(32.0)]);

        assert!(
            Layer::damage(&frame(), &frame()).is_empty(),
            "a frame that drew the same commands has changed nothing"
        );
    }

    #[test]
    fn a_command_that_changed_damages_its_own_rectangle() {
        let damage = Layer::damage(
            &layer(vec![square(0.0), square(32.0)]),
            &layer(vec![square(0.0), square_in(32.0, Color::BLACK)]),
        );

        assert_eq!(
            damage,
            vec![covered(32.0)],
            "only the square whose contents changed"
        );
    }

    #[test]
    fn an_insertion_damages_the_insertion_and_nothing_after_it() {
        // One command appears in the middle of the list. Pairing by index — which is what
        // `iced_tiny_skia` does — would compare every command after it against its neighbour and
        // damage all of them; pairing by geometry stops at the one that actually appeared. This is
        // the launcher's case: 19,733 px this way against 82,027 there.
        let damage = Layer::damage(
            &layer(vec![square(0.0), square(32.0), square(48.0)]),
            &layer(vec![square(0.0), square(16.0), square(32.0), square(48.0)]),
        );

        assert_eq!(damage, vec![covered(16.0)]);
    }

    #[test]
    fn a_command_that_went_away_damages_the_rectangle_it_left() {
        let damage = Layer::damage(
            &layer(vec![square(0.0), square(16.0), square(32.0)]),
            &layer(vec![square(0.0), square(32.0)]),
        );

        assert_eq!(damage, vec![covered(16.0)]);
    }

    #[test]
    fn two_commands_in_one_rectangle_damage_only_that_rectangle() {
        // A fill and its border are two commands with the same bounds. The walk pairs them in list
        // order, so when the two shuffle the rectangle is reported once per position — two entries,
        // the same rectangle. The caller groups them; what matters here is that no *other*
        // rectangle is reported, and that the area does not grow.
        let damage = Layer::damage(
            &layer(vec![square(0.0), square_in(0.0, Color::BLACK)]),
            &layer(vec![
                square_in(0.0, Color::BLACK),
                square_in(0.0, Color::WHITE),
            ]),
        );

        assert!(!damage.is_empty(), "the rectangle has to be reported");
        assert!(
            damage.iter().all(|bounds| *bounds == covered(0.0)),
            "and nothing but that rectangle: {damage:?}"
        );
    }

    #[test]
    fn a_moved_command_damages_both_its_places() {
        let damage = Layer::damage(&layer(vec![square(0.0)]), &layer(vec![square(32.0)]));

        assert_eq!(
            damage,
            vec![covered(0.0), covered(32.0)],
            "where it was and where it went"
        );
    }

    #[test]
    fn a_clipped_command_damages_only_what_the_clip_allowed() {
        let clip = Rectangle::new(Point::new(0.0, 0.0), Size::new(4.0, 64.0));

        let damage = Layer::damage(
            &Layer::with_bounds(clip),
            &Layer {
                quads: vec![square(0.0)],
                ..Layer::with_bounds(clip)
            },
        );

        assert_eq!(
            damage,
            vec![Rectangle::new(Point::new(0.0, 0.0), Size::new(4.0, 9.0))],
            "the clip cuts the command's damage down to what it could paint"
        );
    }

    #[test]
    fn a_layer_that_moved_damages_both_its_places() {
        // `iced_tiny_skia`'s rule, and the same one here: a layer is damage as a whole when its
        // bounds change, because the clip every command was recorded inside is not the same one.
        let damage = Layer::damage(
            &Layer::with_bounds(Rectangle::new(Point::ORIGIN, Size::new(32.0, 32.0))),
            &Layer::with_bounds(Rectangle::new(Point::ORIGIN, Size::new(48.0, 32.0))),
        );

        assert_eq!(
            damage,
            vec![
                Rectangle::new(Point::ORIGIN, Size::new(32.0, 32.0)),
                Rectangle::new(Point::ORIGIN, Size::new(48.0, 32.0))
            ]
        );
    }

    #[test]
    fn a_label_that_changed_damages_its_own_box() {
        let damage = Layer::damage(
            &Layer {
                text: vec![text("10")],
                ..Layer::with_bounds(SCREEN)
            },
            &Layer {
                text: vec![text("11")],
                ..Layer::with_bounds(SCREEN)
            },
        );

        assert_eq!(
            damage,
            vec![Rectangle::new(Point::ORIGIN, Size::new(21.0, 11.0))],
            "the label's own box, grown by the pixel its edge bleeds into"
        );
    }

    #[test]
    fn a_canvas_damages_only_the_command_that_grew() {
        // The claim this renderer exists for, on top of the shape it shares with `iced_tiny_skia`:
        // a canvas that redraws a stroke animation changes one of its commands, and the pieces
        // that are already finished are not reported at all.
        let frame = |tip: f32| canvas(vec![group(vec![square(0.0), square_w(32.0, tip)])]);

        let damage = Layer::damage(&frame(8.0), &frame(16.0));

        assert_eq!(
            damage,
            vec![covered(32.0), covered_w(32.0, 16.0)],
            "the tip: where it was, and where it grew to"
        );
    }

    #[test]
    fn a_cache_that_nobody_cleared_is_unchanged() {
        let commands: Arc<[Primitive]> = Arc::from(vec![square(0.0)]);

        let item = |commands| Item::Cached(commands, SCREEN, Transformation::IDENTITY);

        assert!(
            Layer::damage(
                &canvas(vec![item(commands.clone())]),
                &canvas(vec![item(commands)])
            )
            .is_empty(),
            "the same buffer is the same picture, and it was not even looked into"
        );
    }

    #[test]
    fn a_cache_that_was_rebuilt_damages_only_what_it_changed() {
        let damage = Layer::damage(
            &canvas(vec![cached(vec![square(0.0), square(32.0)])]),
            &canvas(vec![cached(vec![
                square(0.0),
                square_in(32.0, Color::BLACK),
            ])]),
        );

        assert_eq!(
            damage,
            vec![covered(32.0)],
            "a cache that was rebuilt is walked like a group"
        );
    }
}
