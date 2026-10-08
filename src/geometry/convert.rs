//! Conversions between iced canvas types and pomelo-gfx drawing structures.

use iced_core::{Point, Rectangle};
use iced_graphics::geometry::{self, Gradient, Path, Style};
use pomelo_gfx::{
    Color, FillRule, LineCap, LineJoin, LinearGradient, Paint, Path as GfxPath, PathBuilder,
    Point as GfxPoint, Rect, Shader, SpreadMode, Transform,
};

/// A `Rectangle` as the rasteriser states it, if it has any area at all.
pub fn rect(rectangle: Rectangle) -> Option<Rect> {
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
pub fn convert_path(path: &Path, transform: &Transform) -> Option<GfxPath> {
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
pub fn linear_paint(
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
pub fn rule(rule: geometry::fill::Rule) -> FillRule {
    match rule {
        geometry::fill::Rule::NonZero => FillRule::Winding,
        geometry::fill::Rule::EvenOdd => FillRule::EvenOdd,
    }
}

/// iced's line cap, as the rasteriser's. The names are the same on both sides.
pub fn cap(cap: geometry::stroke::LineCap) -> LineCap {
    match cap {
        geometry::stroke::LineCap::Butt => LineCap::Butt,
        geometry::stroke::LineCap::Round => LineCap::Round,
        geometry::stroke::LineCap::Square => LineCap::Square,
    }
}

/// iced's line join, as the rasteriser's. The names are the same on both sides.
pub fn join(join: geometry::stroke::LineJoin) -> LineJoin {
    match join {
        geometry::stroke::LineJoin::Miter => LineJoin::Miter,
        geometry::stroke::LineJoin::Round => LineJoin::Round,
        geometry::stroke::LineJoin::Bevel => LineJoin::Bevel,
    }
}
