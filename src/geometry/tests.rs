use super::*;
use iced_core::Color as IcedColor;
use iced_core::{Point, Size, Vector};
use iced_graphics::cache::{Cached, Group};
use iced_graphics::geometry::fill::Fill;
use iced_graphics::geometry::frame::Backend;
use iced_graphics::geometry::stroke::Stroke;
use iced_graphics::geometry::{self, Gradient, Path, Style};
use iced_graphics::gradient::Linear;
use pomelo_gfx::{Canvas, Pixmap565, Rect, Transform};

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
            crate::renderer::Placement::IDENTITY,
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
            crate::renderer::Placement::IDENTITY,
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

    let point = frame.transform().map_point(pomelo_gfx::Point::from_xy(1.0, 1.0));

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
