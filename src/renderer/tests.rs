    use super::*;
    use iced_core::border::Radius as IcedRadius;
    use iced_core::gradient::Linear;
    use iced_core::renderer::Renderer as _;
    use iced_core::{Border, Color as IcedColor, Size as IcedSize};
    use iced_graphics::geometry::frame::Backend as _;
    use iced_graphics::layer::Layer as _;
    use pomelo_gfx::{rgb565_to_rgb888, Pixmap565};

    const SIZE: u32 = 64;

    /// A renderer with a frame open, which is where every frame starts.
    fn renderer() -> Renderer {
        let mut renderer = Renderer::new(Font::default(), Pixels(16.0));

        renderer.reset(Rectangle::with_size(IcedSize::new(
            SIZE as f32,
            SIZE as f32,
        )));

        renderer
    }

    /// A quad covering `bounds`, with no border and no shadow.
    fn quad(bounds: Rectangle) -> Quad {
        Quad {
            bounds,
            ..Quad::default()
        }
    }

    /// Replays the recording into a fresh panel, all of it damaged.
    fn render(renderer: &Renderer) -> Pixmap565 {
        let mut pixmap = Pixmap565::new(SIZE, SIZE).expect("a pixmap");
        let damage = GfxRect::from_ltrb(0.0, 0.0, SIZE as f32, SIZE as f32);

        let mut canvas = Canvas::new(pixmap.as_mut());
        renderer.draw(&mut canvas, damage);
        drop(canvas);

        pixmap
    }

    /// The frame's widget commands, in the order they were recorded.
    fn quads(renderer: &mut Renderer) -> Vec<Primitive> {
        renderer
            .layers()
            .iter()
            .flat_map(|layer| layer.quads.iter().cloned())
            .collect()
    }

    /// The colour of one pixel.
    fn at(pixmap: &Pixmap565, x: u32, y: u32) -> (u8, u8, u8) {
        rgb565_to_rgb888(pixmap.data()[(y * SIZE + x) as usize])
    }

    fn white_quad() -> Quad {
        quad(Rectangle::new(
            Point::new(8.0, 8.0),
            IcedSize::new(16.0, 16.0),
        ))
    }

    #[test]
    fn a_quad_becomes_a_command_and_paints() {
        let mut renderer = renderer();

        renderer.fill_quad(white_quad(), Background::Color(IcedColor::WHITE));

        assert_eq!(quads(&mut renderer).len(), 1, "one quad is one command");

        let pixmap = render(&renderer);

        assert_eq!(at(&pixmap, 16, 16), (255, 255, 255), "the quad's middle");
        assert_eq!(at(&pixmap, 4, 4), (0, 0, 0), "and nothing outside it");
    }

    #[test]
    fn a_radius_keeps_the_corner_empty() {
        let mut renderer = renderer();
        let mut quad = quad(Rectangle::new(
            Point::new(8.0, 8.0),
            IcedSize::new(32.0, 32.0),
        ));
        quad.border.radius = IcedRadius {
            top_left: 12.0,
            top_right: 12.0,
            bottom_right: 12.0,
            bottom_left: 12.0,
        };

        renderer.fill_quad(quad, Background::Color(IcedColor::WHITE));

        let pixmap = render(&renderer);

        assert_eq!(at(&pixmap, 24, 24), (255, 255, 255), "the middle is filled");
        assert_eq!(
            at(&pixmap, 9, 9),
            (0, 0, 0),
            "the corner the radius cuts off is not"
        );
        assert_eq!(
            at(&pixmap, 20, 10),
            (255, 255, 255),
            "and the edge between the corners is"
        );
    }

    #[test]
    fn a_border_is_a_second_command_along_the_inside_edge() {
        let mut renderer = renderer();
        let mut quad = quad(Rectangle::new(
            Point::new(8.0, 8.0),
            IcedSize::new(32.0, 32.0),
        ));
        quad.border = Border {
            color: IcedColor::from_rgb(1.0, 0.0, 0.0),
            width: 4.0,
            radius: IcedRadius::default(),
        };

        renderer.fill_quad(quad, Background::Color(IcedColor::WHITE));

        assert_eq!(
            quads(&mut renderer).len(),
            2,
            "a fill and a border are two commands"
        );

        let pixmap = render(&renderer);

        assert_eq!(at(&pixmap, 10, 24).0, 255, "the border is red");
        assert_eq!(at(&pixmap, 10, 24).1, 0);
        assert_eq!(
            at(&pixmap, 24, 24),
            (255, 255, 255),
            "and the fill is still under it"
        );
    }

    #[test]
    fn a_gradient_background_arrives_as_a_gradient() {
        let mut renderer = renderer();

        // A quarter turn, so the gradient runs left to right across the quad: the angle is
        // measured from `to_distance`, which is the same rule the picture is drawn by.
        let gradient = Linear::new(iced_core::Radians(std::f32::consts::FRAC_PI_2))
            .add_stop(0.0, IcedColor::from_rgb(1.0, 0.0, 0.0))
            .add_stop(1.0, IcedColor::from_rgb(0.0, 0.0, 1.0));

        renderer.fill_quad(
            quad(Rectangle::new(
                Point::new(8.0, 8.0),
                IcedSize::new(32.0, 32.0),
            )),
            Background::Gradient(iced_core::Gradient::Linear(gradient)),
        );

        let pixmap = render(&renderer);

        let left = at(&pixmap, 10, 24);
        let right = at(&pixmap, 38, 24);

        assert!(left.0 > left.2, "the near end is the first stop: {left:?}");
        assert!(right.2 > right.0, "the far end is the last stop: {right:?}");
    }

    #[test]
    fn a_canvas_geometry_is_one_item_that_holds_its_commands() {
        let mut renderer = renderer();

        let mut frame = geometry::Frame::new(Rectangle::with_size(IcedSize::new(
            SIZE as f32,
            SIZE as f32,
        )));

        use iced_graphics::geometry::frame::Backend;

        frame.fill_rectangle(
            Point::new(0.0, 0.0),
            IcedSize::new(16.0, 16.0),
            IcedColor::WHITE,
        );
        frame.fill_rectangle(
            Point::new(0.0, 32.0),
            IcedSize::new(16.0, 16.0),
            IcedColor::WHITE,
        );

        iced_geometry::Renderer::draw_geometry(&mut renderer, frame.into_geometry());

        let recorded = renderer.layers()[0].primitives.clone();

        assert_eq!(recorded.len(), 1, "a canvas is one item");
        assert_eq!(
            recorded[0].as_slice().len(),
            2,
            "and the commands the canvas drew are inside it"
        );

        let pixmap = render(&renderer);

        assert_eq!(at(&pixmap, 8, 8), (255, 255, 255));
        assert_eq!(at(&pixmap, 8, 40), (255, 255, 255));
        assert_eq!(at(&pixmap, 8, 24), (0, 0, 0), "and nothing between them");
    }

    // The damage itself — how a frame's commands are paired, and what that costs a canvas — is
    // tested where it lives: `layer::tests`.

    #[test]
    fn a_layer_narrows_every_clip_inside_it() {
        let mut renderer = renderer();

        let layer = Rectangle::new(Point::new(0.0, 0.0), IcedSize::new(20.0, SIZE as f32));

        iced_core::renderer::Renderer::start_layer(&mut renderer, layer);
        renderer.fill_quad(
            quad(Rectangle::new(
                Point::new(4.0, 4.0),
                IcedSize::new(40.0, 40.0),
            )),
            Background::Color(IcedColor::WHITE),
        );
        iced_core::renderer::Renderer::end_layer(&mut renderer);

        assert_eq!(
            renderer.layers()[1].bounds,
            layer,
            "the layer is the clip the commands were recorded under"
        );

        let pixmap = render(&renderer);

        assert_eq!(at(&pixmap, 16, 16), (255, 255, 255), "inside the layer");
        assert_eq!(
            at(&pixmap, 30, 16),
            (0, 0, 0),
            "outside the layer, though the quad covers it"
        );
    }

    #[test]
    fn a_command_bounds_are_its_clip() {
        let mut renderer = renderer();
        let quad = white_quad();

        renderer.fill_quad(quad, Background::Color(IcedColor::WHITE));

        let frame = Layer::with_bounds(Rectangle::with_size(IcedSize::new(
            SIZE as f32,
            SIZE as f32,
        )));
        let recorded = renderer.layers()[0].clone();

        assert_eq!(
            Layer::damage(&frame, &recorded),
            vec![Rectangle::new(
                Point::new(7.0, 7.0),
                IcedSize::new(18.0, 18.0)
            )],
            "a command's damage is its own bounds, grown by the edge it bleeds into"
        );
    }

    #[test]
    fn a_canvas_below_the_origin_keeps_its_lower_rows() {
        // A canvas states its own frame as its own size and the widget wraps the draw in a
        // translation, so the two are only comparable once the frame's clip has been placed. A
        // canvas that is not at the origin is where that matters: the launcher puts one under a
        // status bar, 436 px tall on a 480 px screen.
        let mut renderer = renderer();

        let size = Size::new(SIZE as f32, SIZE as f32 - 16.0);
        let offset = 16.0;

        iced_core::renderer::Renderer::with_translation(
            &mut renderer,
            Vector::new(0.0, offset),
            |renderer| {
                let mut frame = geometry::Frame::new(Rectangle::with_size(size));
                frame.fill_rectangle(Point::ORIGIN, size, IcedColor::WHITE);

                iced_geometry::Renderer::draw_geometry(renderer, frame.into_geometry());
            },
        );

        let screen = Layer::with_bounds(Rectangle::with_size(IcedSize::new(
            SIZE as f32,
            SIZE as f32,
        )));

        assert_eq!(
            Layer::damage(&screen, &renderer.layers()[0]),
            vec![Rectangle::new(Point::new(0.0, offset), size)],
            "the whole frame is on the panel, so the whole frame is damageable"
        );
    }

    #[test]
    fn a_reset_clears_the_recording_and_the_layers() {
        let mut renderer = renderer();

        iced_core::renderer::Renderer::start_layer(
            &mut renderer,
            Rectangle::with_size(IcedSize::new(1.0, 1.0)),
        );
        renderer.fill_quad(white_quad(), Background::Color(IcedColor::WHITE));

        iced_core::renderer::Renderer::reset(
            &mut renderer,
            Rectangle::with_size(IcedSize::new(SIZE as f32, SIZE as f32)),
        );

        assert_eq!(
            renderer.layers().len(),
            1,
            "the layer that was open is gone"
        );
        assert!(
            renderer.layers()[0].quads.is_empty(),
            "the recording is empty again"
        );

        renderer.fill_quad(white_quad(), Background::Color(IcedColor::WHITE));

        assert_eq!(
            renderer.layers()[0].bounds,
            Rectangle::with_size(IcedSize::new(SIZE as f32, SIZE as f32)),
            "and the base layer is the frame's bounds"
        );
    }

    #[test]
    fn a_transformation_is_counted_and_changes_nothing() {
        let mut renderer = renderer();

        iced_core::renderer::Renderer::start_transformation(
            &mut renderer,
            Transformation::IDENTITY,
        );

        assert_eq!(renderer.open_transformations(), 1);

        iced_core::renderer::Renderer::end_transformation(&mut renderer);

        assert_eq!(renderer.open_transformations(), 0);
    }
