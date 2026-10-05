//! iced's canvas geometry, recorded in the shape `pomelo-gfx` draws.
//!
//! A `canvas` does not hand its renderer a picture, and it does not hand it a mesh. It hands it
//! *geometry*, through types the renderer defines for itself:
//!
//! ```text
//! trait iced_graphics::geometry::Renderer: core::Renderer {
//!     type Geometry: Cached;
//!     type Frame: frame::Backend<Geometry = Self::Geometry>;
//!     fn new_frame(&self, bounds: Rectangle) -> Self::Frame;
//!     fn draw_geometry(&mut self, geometry: Self::Geometry);
//! }
//! ```
//!
//! So the calls a `canvas` makes arrive as `Frame::fill` and `Frame::stroke`, with a real path and
//! a real paint — `iced_graphics::geometry::{Path, Style}`, the second of which is
//! `Solid(Color) | Gradient(Linear)` — and not as a triangle list. That is the whole reason this
//! module is short: **the paths arrive as paths, and `pomelo-gfx` already draws paths.** What is
//! left is the conversion (lyon → `pomelo_gfx::Path`, `Style` → `pomelo_gfx::Paint`) and the
//! replay into a [`Canvas`], clipped to one damage rectangle.
//!
//! # Transforms are baked, and the stroke width is not scaled
//!
//! `Frame`'s transform is applied to the path's points as each command is recorded, exactly as
//! `iced_tiny_skia` does it, and the gradient's endpoints with it. The stroke width is left as
//! given: it is therefore a **device-space** width, and a `frame.scale(0.5)` half-sizes the path
//! without thinning the line. Matching `iced_tiny_skia` here — rather than scaling the width with
//! the transform, which is what every other canvas does — means the same `canvas` draws the same
//! picture on either renderer, which is what lets a test compare them.
//!
//! # What is not here yet
//!
//! Canvas text. `Frame::fill_text` and `Frame::stroke_text` are `todo!()` rather than silently
//! dropped, because iced's canvas text needs a `Paragraph` and a `blit_mask` replay, and neither
//! the launcher nor hello draws canvas text. When it lands it will be a [`Text`](crate::layer::Text)
//! in the layer's own `text` vector, the way `iced_tiny_skia` records it — which is why there is no
//! text case in [`Primitive`] at all: a widget's label is a text *item*, not a canvas command.

use std::sync::Arc;

use iced_core::{Point, Radians, Rectangle, Size, Vector};
use iced_graphics::cache::{Cached, Group};
use iced_graphics::geometry::fill::Fill;
use iced_graphics::geometry::frame::Backend;
use iced_graphics::geometry::stroke::Stroke;
use iced_graphics::geometry::{self, Gradient, Path, Style, Text};
use iced_graphics::text::paragraph::Weak as ShapedText;

use crate::renderer::Placement;

use pomelo_gfx::{
    Canvas, Color, FillRule, LineCap, LineJoin, LinearGradient, Paint, Path as GfxPath,
    PathBuilder, Point as GfxPoint, RRect, Rect, Shader, SpreadMode, Stroke as GfxStroke,
    Transform,
};

/// One drawing command, in the shape the rasteriser takes it.
///
/// The path is already in device space and the paint already carries a transformed gradient, so a
/// replay is nothing but a call and a clip.
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    /// A filled path.
    Fill {
        /// The path, transformed.
        path: GfxPath,
        /// What to fill it with.
        paint: Paint<'static>,
        /// Which points are inside it.
        rule: FillRule,
    },
    /// A filled rectangle.
    ///
    /// Kept as a rectangle instead of being turned into a four-point path because the rasteriser
    /// has a rectangle filler that samples a shader (`Canvas::fill_rect`), and `fill_path` is not
    /// a substitute for it: that one flattens the path and strokes the outline with a 1 px line.
    Rect {
        /// The rectangle, transformed.
        rect: Rect,
        /// What to fill it with.
        paint: Paint<'static>,
    },
    /// A filled rounded rectangle.
    ///
    /// The paint is a colour in practice, and that is the rasteriser's doing: its rounded-rectangle
    /// filler takes a colour, and its shader-aware filler has no corner geometry. A gradient here
    /// is therefore painted into the bounds and squares the corners. Nothing in this OS draws one.
    Rounded {
        /// The rectangle and its corner radius.
        rrect: RRect,
        /// What to fill it with.
        paint: Paint<'static>,
    },
    /// The outline of a rounded rectangle, inside its bounds.
    RoundedStroke {
        /// The rectangle and its corner radius.
        rrect: RRect,
        /// What to stroke it with.
        paint: Paint<'static>,
        /// How wide the line is.
        width: f32,
    },
    /// A stroked path.
    Stroke {
        /// The path, transformed. The width travels in `stroke`.
        path: GfxPath,
        /// What to stroke it with.
        paint: Paint<'static>,
        /// How wide, and with what ends and joins.
        stroke: GfxStroke,
    },
    /// A 16-bit RGB565 bitmap image with optional 8-bit alpha mask.
    Image565 {
        /// The destination rectangle.
        rect: Rect,
        /// The RGB565 pixels.
        pixels: &'static [u16],
        /// The optional alpha mask.
        alpha: Option<&'static [u8]>,
        /// Source width in pixels.
        src_w: u32,
        /// Source height in pixels.
        src_h: u32,
    },
}

impl Parameters {
    /// The key iced's shaping cache is indexed by.
    ///
    /// `align_y` is not part of it and does not need to be: a vertical alignment moves where the
    /// text is drawn, not what its shape is.
    pub fn key(&self) -> iced_graphics::text::cache::Key<'_> {
        iced_graphics::text::cache::Key {
            content: &self.content,
            size: self.size,
            line_height: self.line_height,
            font: self.font,
            bounds: self.bounds,
            shaping: self.shaping,
            align_x: self.align_x,
        }
    }
}

/// A run of text, in one of the two shapes iced hands it over in.
#[derive(Clone, PartialEq)]
pub enum TextRun {
    /// The parameters of some text, which are shaped when it is drawn.
    ///
    /// `fill_text` receives this: widgets that do not shape for themselves hand over the content
    /// and the geometry, and share one shaping cache through `iced_graphics::text::cache`.
    Parameters(Parameters),
    /// A paragraph someone else shaped, and is holding on to.
    ///
    /// `fill_paragraph` receives one — the `Text` widget shapes its own, in its own tree state, and
    /// re-shapes it only when the text changes. The weak reference is what lets the recording point
    /// at it without owning it.
    Shaped(ShapedText),
}

impl std::fmt::Debug for TextRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextRun::Parameters(parameters) => {
                f.debug_tuple("Parameters").field(parameters).finish()
            }
            // A shaped paragraph has no `Debug` of its own — it holds a cosmic-text buffer — and
            // printing one would only ever be noise in a test failure.
            TextRun::Shaped(_) => f.write_str("Shaped(..)"),
        }
    }
}

/// The text `fill_text` was handed, in the form the shaper needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Parameters {
    /// What it says.
    pub content: String,
    /// How big it is.
    pub size: f32,
    /// How far the lines are apart.
    pub line_height: f32,
    /// Which font.
    pub font: iced_core::Font,
    /// How the lines line up.
    pub align_x: iced_core::text::Alignment,
    /// How the characters are shaped.
    pub shaping: iced_core::text::Shaping,
    /// The size it was laid out into.
    pub bounds: Size,
}

impl Primitive {
    /// The device-space rectangle this command can touch.
    ///
    /// A path's bounds are the bounds of its flattened points, and a stroke's are those grown by
    /// half the width — grown by a whole one, because a round cap and a round join both reach
    /// half a width past the end of the line, and the flattening is a chord approximation of a
    /// curve that bulges outside it.
    pub fn bounds(&self) -> Rectangle {
        match self {
            Primitive::Fill { path, .. } => points_bounds(path),
            Primitive::Rect { rect, .. } => rect_bounds(*rect),
            Primitive::Rounded { rrect, .. } => rect_bounds(rrect.rect),
            Primitive::RoundedStroke { rrect, .. } => rect_bounds(rrect.rect),
            Primitive::Stroke { path, stroke, .. } => points_bounds(path).expand(stroke.width),
            Primitive::Image565 { rect, .. } => rect_bounds(*rect),
        }
    }
}

/// A rasteriser rectangle as one iced can measure.
fn rect_bounds(rect: Rect) -> Rectangle {
    Rectangle::new(
        Point::new(rect.left(), rect.top()),
        Size::new(rect.width, rect.height),
    )
}

/// The bounds of a path's flattened points.
fn points_bounds(path: &GfxPath) -> Rectangle {
    path.flatten(0.5)
        .into_iter()
        .flatten()
        .fold(None, |bounds: Option<Rectangle>, point| {
            let point = Rectangle {
                x: point.x,
                y: point.y,
                width: 0.0,
                height: 0.0,
            };

            Some(match bounds {
                Some(bounds) => bounds.union(&point),
                None => point,
            })
        })
        .unwrap_or(Rectangle::INFINITE)
}

/// A frame of canvas geometry, in the shape `pomelo-gfx` draws it.
///
/// This is `geometry::Renderer::Frame`. A `canvas` builds one, calls `fill` and `stroke` on it,
/// and hands the result back as a [`Geometry`].
#[derive(Debug, Clone)]
pub struct Frame {
    clip_bounds: Rectangle,
    transform: Transform,
    stack: Vec<Transform>,
    primitives: Vec<Primitive>,
}

impl Frame {
    /// A frame that records into `bounds`, with no transform.
    pub fn new(bounds: Rectangle) -> Self {
        Self {
            clip_bounds: bounds,
            transform: Transform::identity(),
            stack: Vec::new(),
            primitives: Vec::new(),
        }
    }

    /// The frame's transform, as the rasteriser states it.
    pub fn transform(&self) -> Transform {
        self.transform
    }
}

/// Composes a translation into a frame's transform, on the **right**.
///
/// `self * T` and not `T * self`, which is the whole of the difference between the two ways a canvas
/// can mean "translate this frame": a point `p` is drawn at `p + translation`, in the frame's own
/// coordinates, whatever the transform already in force is.
///
/// This is what iced's own backends do -- `iced_tiny_skia`'s canvas frame composes every one of its
/// four transform operations with `pre_concat`, and `pre_concat` there is a left product. It matters
/// for the one thing a canvas does with two transforms in a row: after `scale(s)` and then
/// `translate(o)`, a point is drawn at `(p + o) * s`, where the other product would put it at
/// `p * s + o`. The signature's numbers decide the question on their own: its origin is 357 canonical
/// units to the left of the canvas's top-left corner, so the wrong order draws the whole animation
/// below the bottom edge of the panel.
fn compose_translation(transform: Transform, translation: Vector) -> Transform {
    Transform::from_row(
        transform.sx,
        transform.ky,
        transform.kx,
        transform.sy,
        transform.sx * translation.x + transform.kx * translation.y + transform.tx,
        transform.ky * translation.x + transform.sy * translation.y + transform.ty,
    )
}

/// Composes a scale into a frame's transform, on the right.
///
/// The linear part is scaled and the translation is left where it was, so a scale composes with a
/// translation that is already in force rather than replacing it.
///
/// The cases are pinned below against `tiny_skia`'s own, which is the implementation this has to
/// agree with: `Transform::from_row(1.2, 3.4, -5.6, -7.8, 1.2, 3.4).pre_scale(2.0, -4.0)` is
/// `from_row(2.4, 6.8, 22.4, 31.2, 1.2, 3.4)`, and `post_scale` of the same pair is
/// `from_row(2.4, -13.6, -11.2, 31.2, 2.4, -13.6)`.
fn compose_scale(transform: Transform, x: f32, y: f32) -> Transform {
    Transform::from_row(
        transform.sx * x,
        transform.ky * x,
        transform.kx * y,
        transform.sy * y,
        transform.tx,
        transform.ty,
    )
}

/// Composes a rotation into a frame's transform, on the right.
fn compose_rotation(transform: Transform, radians: f32) -> Transform {
    let (sin, cos) = radians.sin_cos();

    Transform::from_row(
        transform.sx * cos + transform.kx * sin,
        transform.ky * cos + transform.sy * sin,
        transform.kx * cos - transform.sx * sin,
        transform.sy * cos - transform.ky * sin,
        transform.tx,
        transform.ty,
    )
}

impl Backend for Frame {
    type Geometry = Geometry;

    fn width(&self) -> f32 {
        self.clip_bounds.width
    }

    fn height(&self) -> f32 {
        self.clip_bounds.height
    }

    fn size(&self) -> Size {
        self.clip_bounds.size()
    }

    fn center(&self) -> Point {
        self.clip_bounds.center()
    }

    fn push_transform(&mut self) {
        self.stack.push(self.transform);
    }

    fn pop_transform(&mut self) {
        if let Some(transform) = self.stack.pop() {
            self.transform = transform;
        }
    }

    fn translate(&mut self, translation: Vector) {
        self.transform = compose_translation(self.transform, translation);
    }

    fn rotate(&mut self, angle: impl Into<Radians>) {
        self.transform = compose_rotation(self.transform, angle.into().0);
    }

    fn scale(&mut self, scale: impl Into<f32>) {
        let scale = scale.into();

        self.transform = compose_scale(self.transform, scale, scale);
    }

    fn scale_nonuniform(&mut self, scale: impl Into<Vector>) {
        let scale = scale.into();

        self.transform = compose_scale(self.transform, scale.x, scale.y);
    }

    /// Records a path filled with a colour or a gradient.
    fn fill(&mut self, path: &Path, fill: impl Into<Fill>) {
        let fill = fill.into();

        let Some(path) = convert_path(path, &self.transform) else {
            return;
        };

        self.primitives.push(Primitive::Fill {
            path,
            paint: paint_of(fill.style, &self.transform),
            rule: rule(fill.rule),
        });
    }

    /// Records a filled rectangle.
    ///
    /// A rectangle stays a rectangle when the transform only translates and scales it, which is
    /// the case for every canvas that draws paper rather than dials. Once the transform can turn a
    /// corner — a rotation, a skew — the four corners stop being axis-aligned and there is no
    /// rectangle left to hand the filler, so it goes down the path route instead.
    fn fill_rectangle(&mut self, top_left: Point, size: Size, fill: impl Into<Fill>) {
        let fill = fill.into();

        if self.transform.kx != 0.0 || self.transform.ky != 0.0 {
            self.fill(&Path::rectangle(top_left, size), fill);

            return;
        }

        let rect = self.transform.map_rect(Rect::from_ltwh(
            top_left.x,
            top_left.y,
            size.width,
            size.height,
        ));

        self.primitives.push(Primitive::Rect {
            rect,
            paint: paint_of(fill.style, &self.transform),
        });
    }

    /// Records a stroked path.
    fn stroke<'a>(&mut self, path: &Path, stroke: impl Into<Stroke<'a>>) {
        let stroke = stroke.into();

        let Some(path) = convert_path(path, &self.transform) else {
            return;
        };

        self.primitives.push(Primitive::Stroke {
            path,
            paint: paint_of(stroke.style, &self.transform),
            stroke: GfxStroke {
                width: stroke.width,
                line_cap: cap(stroke.line_cap),
                line_join: join(stroke.line_join),
                ..GfxStroke::default()
            },
        });
    }

    /// Records a stroked rectangle, which is a path like any other here.
    fn stroke_rectangle<'a>(&mut self, top_left: Point, size: Size, stroke: impl Into<Stroke<'a>>) {
        self.stroke(&Path::rectangle(top_left, size), stroke);
    }

    /// Not recorded yet: see the module docs. A `canvas` that draws its own text will hit this.
    fn fill_text(&mut self, _text: impl Into<Text>) {
        todo!("canvas text: a Paragraph, and a blit_mask replay")
    }

    /// Not recorded yet: see the module docs.
    fn stroke_text<'a>(&mut self, _text: impl Into<Text>, _stroke: impl Into<Stroke<'a>>) {
        todo!("canvas stroked text: same as fill_text, with an outline to fill")
    }

    /// A frame for a group drawn into `clip_bounds`.
    ///
    /// The bounds intersect with the enclosing frame's, so a group can only ever narrow the clip.
    fn draft(&mut self, clip_bounds: Rectangle) -> Self {
        Self {
            clip_bounds: self
                .clip_bounds
                .intersection(&clip_bounds)
                .unwrap_or(Rectangle::INFINITE),
            transform: self.transform,
            stack: Vec::new(),
            primitives: Vec::new(),
        }
    }

    /// Appends a group's commands to this frame's.
    fn paste(&mut self, frame: Self) {
        self.primitives.extend(frame.primitives);
    }

    /// Not recorded yet: images need `iced_graphics`' `image` feature and the `image` crate, and
    /// they are the same `allocate_image` hole `Renderer` already reports as `Unsupported`.
    fn draw_image(&mut self, _bounds: Rectangle, _image: impl Into<geometry::Image>) {}

    /// Not recorded yet: as `draw_image`, for the `svg` feature.
    fn draw_svg(&mut self, _bounds: Rectangle, _svg: impl Into<geometry::Svg>) {}

    fn into_geometry(self) -> Self::Geometry {
        Geometry::Live {
            primitives: self.primitives,
            clip_bounds: self.clip_bounds,
        }
    }
}

/// A recorded frame of canvas geometry.
#[derive(Debug, Clone)]
pub enum Geometry {
    /// Recorded this frame.
    Live {
        /// What was drawn, in order.
        primitives: Vec<Primitive>,
        /// What it was clipped to.
        clip_bounds: Rectangle,
    },
    /// Kept from an earlier frame by a `canvas::Cache`.
    Cache(Cache),
}

/// A [`Geometry`] that a `canvas::Cache` is holding on to.
#[derive(Debug, Clone)]
pub struct Cache {
    /// What was drawn, in order.
    pub primitives: Arc<[Primitive]>,
    /// What it was clipped to.
    pub clip_bounds: Rectangle,
}

impl Geometry {
    /// What was drawn, in order.
    pub fn primitives(&self) -> &[Primitive] {
        match self {
            Geometry::Live { primitives, .. } => primitives,
            Geometry::Cache(cache) => &cache.primitives,
        }
    }

    /// What it was clipped to.
    pub fn clip_bounds(&self) -> Rectangle {
        match self {
            Geometry::Live { clip_bounds, .. } => *clip_bounds,
            Geometry::Cache(cache) => cache.clip_bounds,
        }
    }
}

impl Cached for Geometry {
    type Cache = Cache;

    fn load(cache: &Cache) -> Self {
        Self::Cache(cache.clone())
    }

    fn cache(self, _group: Group, _previous: Option<Cache>) -> Cache {
        match self {
            Self::Live {
                primitives,
                clip_bounds,
            } => Cache {
                primitives: Arc::from(primitives),
                clip_bounds,
            },
            Self::Cache(cache) => cache,
        }
    }
}

/// Draws one command, clipped to `damage`.
///
/// The clip is the point of the whole renderer: `pomelo-gfx`'s rasteriser takes it into the scan,
/// so a command that is asked to paint a rectangle it mostly misses costs what the intersection
/// costs and not what the command covers.
///
/// `placement` is applied *after* the clip, and the order matters: setting the canvas's clip maps
/// the rectangle through whatever transform is in force, so a transformed command would have its
/// clip transformed twice.
pub fn draw(
    canvas: &mut Canvas<'_>,
    primitive: &Primitive,
    clip_bounds: Rectangle,
    damage: Rect,
    placement: Placement,
) {
    let Some(clip) = clip_bounds
        .intersection(&Rectangle {
            x: damage.left(),
            y: damage.top(),
            width: damage.width,
            height: damage.height,
        })
        .and_then(rect)
    else {
        return;
    };

    canvas.save();
    canvas.clip_rect(clip);
    place(canvas, placement);

    match primitive {
        Primitive::Fill { path, paint, rule } => canvas.fill_path(path, paint, *rule),
        // A solid rectangle takes the rasteriser's axis-aligned filler -- the same one every
        // rounded corner and label background goes through -- and only a shader needs the quad
        // filler, which walks the edges to know what to sample.
        Primitive::Rect { rect, paint } => match paint.shader {
            Shader::SolidColor(color) => canvas.draw_rect(*rect, color),
            _ => canvas.fill_rect(*rect, paint),
        },
        Primitive::Rounded { rrect, paint } => match paint.shader {
            Shader::SolidColor(color) => canvas.draw_rrect(*rrect, color),
            _ => canvas.fill_rect(rrect.rect, paint),
        },
        Primitive::RoundedStroke {
            rrect,
            paint,
            width,
        } => {
            // A gradient border draws nothing rather than something wrong: the rasteriser's
            // rounded-rectangle stroke takes a colour.
            if let Shader::SolidColor(color) = paint.shader {
                canvas.draw_rrect_stroke(*rrect, color, *width);
            }
        }
        Primitive::Stroke {
            path,
            paint,
            stroke,
        } => canvas.stroke_path(path, paint, stroke),
        Primitive::Image565 {
            rect,
            pixels,
            alpha,
            src_w,
            src_h,
        } => {
            let tx = rect.left().round() as i32;
            let ty = rect.top().round() as i32;
            let w = rect.width.round() as u32;
            let h = rect.height.round() as u32;
            if let Some(alpha) = alpha {
                if *src_w == w && *src_h == h {
                    canvas.blit_image_565_with_alpha(tx, ty, w, h, pixels, alpha);
                } else {
                    canvas.blit_image_565_with_alpha_scaled(
                        tx, ty, w, h, *src_w, *src_h, pixels, alpha,
                    );
                }
            } else {
                canvas.blit_image_565(tx, ty, w, h, pixels);
            }
        }
    }

    canvas.restore();
}

/// Puts a canvas into a command's placement.
///
/// Scale first and then translate, because the rasteriser composes both in the canvas's own space:
/// `scale` then `translate` maps `p` to `p * scale + translation`, which is what the placement means.
pub fn place(canvas: &mut Canvas<'_>, placement: Placement) {
    if placement.is_identity() {
        return;
    }

    canvas.scale(placement.scale, placement.scale);
    canvas.translate(placement.translation.x, placement.translation.y);
}

/// A `Rectangle` as the rasteriser states it, if it has any area at all.
fn rect(rectangle: Rectangle) -> Option<Rect> {
    (rectangle.width > 0.0 && rectangle.height > 0.0).then(|| {
        Rect::from_ltrb(
            rectangle.x,
            rectangle.y,
            rectangle.x + rectangle.width,
            rectangle.y + rectangle.height,
        )
    })
}

/// Converts a canvas path into one `pomelo-gfx` draws, transformed as it goes.
///
/// A path is a list of verbs; `lyon` hands them out as subpath events, and the builder takes them
/// back one at a time. Nothing here is approximated: a cubic arrives as a cubic.
fn convert_path(path: &Path, transform: &Transform) -> Option<GfxPath> {
    let mut builder = PathBuilder::new();
    let mut drew = false;

    for event in path.raw().iter() {
        match event {
            lyon_path::PathEvent::Begin { at, .. } => {
                let at = transform.map_point(point(&at));
                builder.move_to(at.x, at.y);
            }
            lyon_path::PathEvent::Line { to, .. } => {
                let to = transform.map_point(point(&to));
                builder.line_to(to.x, to.y);
            }
            lyon_path::PathEvent::Quadratic { ctrl, to, .. } => {
                let ctrl = transform.map_point(point(&ctrl));
                let to = transform.map_point(point(&to));
                builder.quad_to(ctrl.x, ctrl.y, to.x, to.y);
            }
            lyon_path::PathEvent::Cubic {
                ctrl1, ctrl2, to, ..
            } => {
                let ctrl1 = transform.map_point(point(&ctrl1));
                let ctrl2 = transform.map_point(point(&ctrl2));
                let to = transform.map_point(point(&to));
                builder.cubic_to(ctrl1.x, ctrl1.y, ctrl2.x, ctrl2.y, to.x, to.y);
            }
            lyon_path::PathEvent::End { .. } => builder.close(),
        }

        drew = true;
    }

    if drew {
        builder.finish()
    } else {
        None
    }
}

/// Converts a canvas paint into one `pomelo-gfx` draws.
///
/// A linear gradient is stated in the same terms on both sides — two endpoints and a list of
/// stops — so this is a translation and not an approximation. Its endpoints are transformed with
/// the path, which is how the gradient stays attached to the shape it is filling.
///
/// A radial gradient does not exist in `iced_graphics` 0.14 (`Gradient` has one variant), so
/// there is nothing to approximate either.
pub fn paint_of(style: Style, transform: &Transform) -> Paint<'static> {
    match style {
        Style::Solid(color) => solid_paint(color),
        Style::Gradient(Gradient::Linear(linear)) => linear_paint(
            linear.start,
            linear.end,
            linear
                .stops
                .iter()
                .flatten()
                .map(|stop| (stop.offset, stop.color)),
            transform,
        ),
    }
}

/// A linear gradient, whatever iced called it, as the rasteriser states it.
///
/// iced has **two** gradients and they are not the same type. `iced_core`'s is what a widget's
/// background is (`Background::Gradient`); `iced_graphics`' is what a canvas fill is
/// (`geometry::Style::Gradient`) — the latter is a `Pod` type with `f16` stops, made to be packed
/// into a vertex buffer. They state the same thing, so they converge here instead of twice above.
fn linear_paint(
    start: Point,
    end: Point,
    stops: impl Iterator<Item = (f32, iced_core::Color)>,
    transform: &Transform,
) -> Paint<'static> {
    let start = transform.map_point(GfxPoint::from_xy(start.x, start.y));
    let end = transform.map_point(GfxPoint::from_xy(end.x, end.y));
    let stops: Vec<pomelo_gfx::GradientStop> = stops
        .map(|(offset, color)| pomelo_gfx::GradientStop::new(offset, color_of(color)))
        .collect();

    // A gradient with no stops at all is transparent, which is what iced draws for one too.
    let fallback = stops
        .first()
        .map(|stop| stop.color)
        .unwrap_or(Color::TRANSPARENT);

    let shader = LinearGradient::new(start, end, stops, SpreadMode::Pad, Transform::identity())
        .unwrap_or(Shader::SolidColor(fallback));

    // The rasteriser does not read `anti_alias` yet — its edges are hard — but it is the honest
    // value to record, and it is what the field is for.
    Paint {
        shader,
        anti_alias: true,
    }
}

/// iced's colour, as the rasteriser's.
pub fn color_of(color: iced_core::Color) -> Color {
    let [r, g, b, a] = color.into_rgba8();

    Color::from_rgba8(r, g, b, a)
}

/// A solid paint, which is what every quad with a colour background becomes.
pub fn solid_paint(color: iced_core::Color) -> Paint<'static> {
    Paint {
        shader: Shader::SolidColor(color_of(color)),
        anti_alias: true,
    }
}

/// A widget background's gradient as the rasteriser states it.
///
/// A quad's background is a gradient in exactly the same sense a canvas fill is, and it goes
/// through the same conversion -- endpoints and stops, no approximation. What differs is how the
/// endpoints are *stated*: a canvas fill gives two points, and a widget background gives an angle,
/// because a background is relative to the shape it fills. `Radians::to_distance` is iced's own
/// angle-to-endpoints rule, so this lands on the same line iced's other backends draw.
pub fn gradient_paint(
    gradient: iced_core::gradient::Gradient,
    bounds: Rectangle,
) -> Paint<'static> {
    match gradient {
        iced_core::gradient::Gradient::Linear(linear) => {
            let (start, end) = linear.angle.to_distance(&bounds);

            linear_paint(
                start,
                end,
                linear
                    .stops
                    .iter()
                    .flatten()
                    .map(|stop| (stop.offset, stop.color)),
                &Transform::identity(),
            )
        }
    }
}

/// A `lyon` point as the rasteriser's.
fn point(point: &lyon_path::math::Point) -> GfxPoint {
    GfxPoint::from_xy(point.x, point.y)
}

/// iced's fill rule, as the rasteriser's. The names are the same on both sides.
fn rule(rule: geometry::fill::Rule) -> FillRule {
    match rule {
        geometry::fill::Rule::NonZero => FillRule::Winding,
        geometry::fill::Rule::EvenOdd => FillRule::EvenOdd,
    }
}

/// iced's line cap, as the rasteriser's. The names are the same on both sides.
fn cap(cap: geometry::stroke::LineCap) -> LineCap {
    match cap {
        geometry::stroke::LineCap::Butt => LineCap::Butt,
        geometry::stroke::LineCap::Round => LineCap::Round,
        geometry::stroke::LineCap::Square => LineCap::Square,
    }
}

/// iced's line join, as the rasteriser's. The names are the same on both sides.
fn join(join: geometry::stroke::LineJoin) -> LineJoin {
    match join {
        geometry::stroke::LineJoin::Miter => LineJoin::Miter,
        geometry::stroke::LineJoin::Round => LineJoin::Round,
        geometry::stroke::LineJoin::Bevel => LineJoin::Bevel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_core::Color as IcedColor;
    use iced_graphics::gradient::Linear;
    use pomelo_gfx::Pixmap565;

    const SIZE: u32 = 64;

    /// A frame the size of the test panel.
    fn frame() -> Frame {
        Frame::new(Rectangle::with_size(Size::new(SIZE as f32, SIZE as f32)))
    }

    /// Replays `geometry` into a fresh panel, clipping each command to all of it.
    fn render(geometry: &Geometry) -> Pixmap565 {
        let mut pixmap = Pixmap565::new(SIZE, SIZE).expect("a pixmap");
        let damage = Rect::from_ltrb(0.0, 0.0, SIZE as f32, SIZE as f32);

        let mut canvas = Canvas::new(pixmap.as_mut());

        for primitive in geometry.primitives() {
            draw(
                &mut canvas,
                primitive,
                geometry.clip_bounds(),
                damage,
                Placement::IDENTITY,
            );
        }

        drop(canvas);

        pixmap
    }

    /// The pixel at `(x, y)`.
    fn at(pixmap: &Pixmap565, x: u32, y: u32) -> u16 {
        pixmap.data()[(y * SIZE + x) as usize]
    }

    #[test]
    fn a_gradient_rectangle_arrives_as_a_gradient() {
        let mut frame = frame();

        let gradient = Linear::new(Point::new(0.0, 0.0), Point::new(SIZE as f32, 0.0))
            .add_stop(0.0, IcedColor::from_rgb(1.0, 0.0, 0.0))
            .add_stop(1.0, IcedColor::from_rgb(0.0, 0.0, 1.0));

        frame.fill_rectangle(
            Point::new(0.0, 0.0),
            Size::new(SIZE as f32, 8.0),
            Fill {
                style: Style::Gradient(Gradient::Linear(gradient)),
                rule: geometry::fill::Rule::NonZero,
            },
        );

        let pixmap = render(&frame.into_geometry());

        let left = pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 1, 4));
        let right = pomelo_gfx::rgb565_to_rgb888(at(&pixmap, SIZE - 2, 4));

        assert!(left.0 > left.2, "the near end is the first stop: {left:?}");
        assert!(right.2 > right.0, "the far end is the last stop: {right:?}");
    }

    #[test]
    #[ignore = "pomelo-gfx has no path filling: `Canvas::fill_path` strokes the outline with a \
                1 px line and ignores the fill rule"]
    fn a_filled_path_is_filled() {
        let mut frame = frame();

        frame.fill(
            &Path::rectangle(Point::new(8.0, 8.0), Size::new(16.0, 16.0)),
            IcedColor::WHITE,
        );

        let pixmap = render(&frame.into_geometry());

        assert_eq!(
            at(&pixmap, 16, 16),
            pomelo_gfx::Color::WHITE.to_rgb565(),
            "the middle of a filled rectangle is filled"
        );
    }

    #[test]
    fn the_damage_rectangle_is_what_limits_the_paint() {
        let mut frame = frame();

        frame.stroke(
            &Path::line(Point::new(4.0, 32.0), Point::new(60.0, 32.0)),
            Stroke {
                style: Style::Solid(IcedColor::WHITE),
                width: 8.0,
                line_cap: geometry::stroke::LineCap::Butt,
                line_join: geometry::stroke::LineJoin::Miter,
                line_dash: Default::default(),
            },
        );

        let geometry = frame.into_geometry();

        let mut pixmap = Pixmap565::new(SIZE, SIZE).expect("a pixmap");
        let mut canvas = Canvas::new(pixmap.as_mut());

        // The left half of the line, and nothing else, is what this frame was asked to repaint --
        // which is the whole point of the renderer.
        let damage = Rect::from_ltrb(0.0, 0.0, 20.0, SIZE as f32);

        for primitive in geometry.primitives() {
            draw(
                &mut canvas,
                primitive,
                geometry.clip_bounds(),
                damage,
                Placement::IDENTITY,
            );
        }

        drop(canvas);

        assert_ne!(
            pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 16, 32)),
            (0, 0, 0),
            "the damaged half is painted"
        );
        assert_eq!(
            pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 40, 32)),
            (0, 0, 0),
            "the undamaged half is not, even though the command covers it"
        );
    }

    #[test]
    fn a_stroke_arrives_with_its_width_cap_and_gradient() {
        let mut frame = frame();

        let gradient = Linear::new(Point::new(8.0, 32.0), Point::new(56.0, 32.0))
            .add_stop(0.0, IcedColor::from_rgb(1.0, 0.0, 0.0))
            .add_stop(1.0, IcedColor::from_rgb(0.0, 0.0, 1.0));

        frame.stroke(
            &Path::line(Point::new(8.0, 32.0), Point::new(56.0, 32.0)),
            Stroke {
                style: Style::Gradient(Gradient::Linear(gradient)),
                width: 8.0,
                line_cap: geometry::stroke::LineCap::Round,
                line_join: geometry::stroke::LineJoin::Round,
                line_dash: Default::default(),
            },
        );

        let geometry = frame.into_geometry();
        let pixmap = render(&geometry);

        // On the line, in the middle: neither endpoint's colour, because the gradient is between
        // them. The assertion is that the shader arrived at all -- a solid fallback would be one
        // endpoint's red or blue.
        let middle = pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 32, 32));

        assert!(
            middle.0 > 40 && middle.2 > 40,
            "the middle of the stroke is a mix of both stops, not either one: {middle:?}"
        );

        // Across the line: 8 px wide, so 4 px out from the middle is off it.
        assert_eq!(
            pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 32, 40)).0,
            0,
            "the stroke is 8 px wide and no more"
        );

        // Past the end of the line: a round cap reaches half a width beyond it. The colour there
        // is the gradient's last stop -- blue -- so it is the blue channel that has to be lit.
        assert_ne!(
            pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 58, 32)),
            (0, 0, 0),
            "the round cap reaches past the end of the line"
        );
    }

    #[test]
    fn a_scale_composes_the_way_tiny_skia_composes_it() {
        let transform = Transform::from_row(1.2, 3.4, -5.6, -7.8, 1.2, 3.4);

        assert_eq!(
            compose_scale(transform, 2.0, -4.0),
            Transform::from_row(2.4, 6.8, 22.4, 31.2, 1.2, 3.4),
            "tiny_skia's `pre_scale` test, verbatim"
        );
        assert_eq!(
            transform.post_scale(2.0, -4.0),
            Transform::from_row(2.4, -13.6, -11.2, 31.2, 2.4, -13.6),
            "and its `post_scale`, which is the one pomelo-gfx offers and this does not want"
        );
    }

    #[test]
    fn a_scale_and_a_translation_compose_in_that_order() {
        // The signature's own shape of a transform: a scale to fit the artwork to the panel, and a
        // translation to centre it, both stated in canonical units.
        let mut frame = frame();
        frame.scale(2.0f32);
        frame.translate(Vector::new(3.0, 4.0));

        let point = frame.transform().map_point(GfxPoint::from_xy(1.0, 1.0));

        assert_eq!(
            (point.x, point.y),
            (8.0, 10.0),
            "(1 + 3) * 2 and (1 + 4) * 2; the other order would give 5 and 6"
        );
    }

    #[test]
    fn a_transform_moves_the_path_but_not_the_width() {
        let mut frame = frame();
        frame.scale(2.0f32);

        frame.stroke(
            &Path::line(Point::new(4.0, 16.0), Point::new(12.0, 16.0)),
            Stroke {
                style: Style::Solid(IcedColor::WHITE),
                width: 4.0,
                line_cap: geometry::stroke::LineCap::Butt,
                line_join: geometry::stroke::LineJoin::Miter,
                line_dash: Default::default(),
            },
        );

        let pixmap = render(&frame.into_geometry());

        assert_ne!(
            pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 16, 32)),
            (0, 0, 0),
            "the line is drawn at twice its position and twice its length"
        );
        assert_eq!(
            pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 16, 38)),
            (0, 0, 0),
            "and its width is the 4 px it was given, not the 8 a scaled width would give"
        );
        assert_eq!(
            pomelo_gfx::rgb565_to_rgb888(at(&pixmap, 6, 32)),
            (0, 0, 0),
            "and the untransformed start is empty"
        );
    }

    #[test]
    fn a_commands_bounds_contain_what_it_paints() {
        let mut frame = frame();

        frame.stroke(
            &Path::line(Point::new(10.0, 10.0), Point::new(40.0, 10.0)),
            Stroke {
                style: Style::Solid(IcedColor::WHITE),
                width: 6.0,
                line_cap: geometry::stroke::LineCap::Butt,
                line_join: geometry::stroke::LineJoin::Miter,
                line_dash: Default::default(),
            },
        );

        let geometry = frame.into_geometry();
        let bounds = geometry.primitives()[0].bounds();
        let pixmap = render(&geometry);

        for y in 0..SIZE {
            for x in 0..SIZE {
                if pomelo_gfx::rgb565_to_rgb888(at(&pixmap, x, y)).0 > 0 {
                    assert!(
                        bounds.contains(Point::new(x as f32 + 0.5, y as f32 + 0.5)),
                        "({x}, {y}) is painted outside the bounds {bounds:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_cache_keeps_the_same_commands() {
        let mut frame = frame();

        frame.fill_rectangle(Point::new(0.0, 0.0), Size::new(8.0, 8.0), IcedColor::WHITE);

        let geometry = frame.into_geometry();
        let primitives = geometry.primitives().to_vec();

        let cache = geometry.cache(Group::unique(), None);
        let loaded = Geometry::load(&cache);

        assert_eq!(
            loaded.primitives(),
            primitives.as_slice(),
            "a cached geometry is the same commands it was cached from"
        );
    }
}
