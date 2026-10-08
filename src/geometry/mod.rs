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
use iced_graphics::geometry::{self, Path, Text};

use crate::renderer::Placement;

use pomelo_gfx::{Canvas, Rect, Shader, Stroke as GfxStroke, Transform};

pub mod convert;
pub mod primitive;

#[cfg(test)]
mod tests;

pub use convert::*;
pub use primitive::*;

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
