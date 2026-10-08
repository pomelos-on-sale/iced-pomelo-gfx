//! Canvas drawing primitives and text runs.

use iced_core::{Point, Rectangle, Size};
use iced_graphics::text::paragraph::Weak as ShapedText;
use pomelo_gfx::{FillRule, Paint, Path as GfxPath, Rect, RRect, Stroke as GfxStroke};

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
