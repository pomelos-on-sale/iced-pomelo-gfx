//! iced's renderer contract, implemented over `pomelo-gfx`.
//!
//! Every pixel this renderer produces is drawn straight into the panel's RGB565 buffer, inside
//! the damaged rectangles and nowhere else. That is the whole reason it exists: `tiny-skia`
//! rasterises a primitive over its full extent and uses the clip mask only to reject pixels at
//! blend time, so a launcher frame that changes one label costs the same as one that redraws the
//! screen — measured on this board at 59 ms for 8,580 px of damage against 377 ms for the whole
//! screen, with 4.5 µs/px on the float path. `pomelo-gfx`'s rasteriser takes the clip into the
//! scan itself, so its cost follows the damage.
//!
//! What is *not* here is the part that made this look expensive the first time: the text engine.
//! `iced_graphics::text::{Paragraph, Editor}` are renderer-independent — cosmic-text shaping,
//! wrapping, hit testing, editing, cached per content — and both of iced's own backends use them
//! as their associated types. The earlier estimate of "19 methods for `Paragraph` plus 14 for
//! `Editor`" was counting code that already exists in a crate we depend on.
//!
//! The contract is fourteen methods, enumerated by the compiler rather than by a document:
//!
//! | trait | methods |
//! |---|---|
//! | `core::Renderer` | `start_layer`, `end_layer`, `start_transformation`, `end_transformation`, `fill_quad`, `reset`, `allocate_image` |
//! | `text::Renderer` | `default_font`, `default_size`, `fill_paragraph`, `fill_editor`, `fill_text` |
//! | `mesh::Renderer` | `draw_mesh`, `draw_mesh_cache` |
//!
//! What is left around them is the recording (our own layer of quads and glyph runs, diffable for
//! damage) and the replay into `pomelo_gfx::Canvas`.
//!
//! This module is behind the `renderer` feature, off by default, until it can draw the launcher.

use std::borrow::Cow;
use std::cell::RefCell;

use iced_core::image;
use iced_core::renderer::Quad;
use iced_core::text;
use iced_core::{Background, Color, Font, Pixels, Point, Rectangle, Size, Transformation, Vector};
use iced_graphics::geometry as iced_geometry;
use iced_graphics::mesh;
use pomelo_gfx::{Canvas, Pixmap565, RRect, Radius, Rect as GfxRect};

use crate::geometry::{self, Parameters, Primitive, TextRun};

/// One recorded command, and the clip it was drawn under.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    /// What to draw.
    pub primitive: Primitive,
    /// The rectangle it may draw inside: its layer's bounds, narrowed by every layer above it.
    pub clip: Rectangle,
    /// Where it was drawn, if it was drawn inside a transformation.
    pub placement: Placement,
}

/// The flat part of a transformation: how far it moves, and how much it scales.
///
/// iced's `Transformation` is a `Mat4`, but what the 2D rasteriser needs from it is what its own
/// backends take — `scale_factor()` and `translation()` — and that is what this keeps. A rotation
/// or a skew would be lost, and nothing on this path produces one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    /// How far the command moved.
    pub translation: Vector,
    /// How much it was scaled.
    pub scale: f32,
}

impl Placement {
    /// A command that was drawn exactly where it said.
    pub const IDENTITY: Self = Self {
        translation: Vector::new(0.0, 0.0),
        scale: 1.0,
    };

    /// Whether this placement leaves a command where it says it is.
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// A rectangle in this placement's coordinates, in the destination's.
    fn map(&self, bounds: Rectangle) -> Rectangle {
        Rectangle {
            x: bounds.x * self.scale + self.translation.x,
            y: bounds.y * self.scale + self.translation.y,
            width: bounds.width * self.scale,
            height: bounds.height * self.scale,
        }
    }
}

impl Default for Placement {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Item {
    /// The device-space area this command can have changed.
    ///
    /// The intersection and not the whole primitive: a command asked to paint a screen-sized
    /// rectangle inside a tile-sized clip can only have changed the tile.
    pub fn bounds(&self) -> Rectangle {
        self.placement
            .map(self.primitive.bounds())
            .intersection(&self.clip)
            .unwrap_or(Rectangle::with_size(Size::ZERO))
    }
}

/// The renderer iced drives.
///
/// It **records** rather than draws. Every call iced makes appends a [`Primitive`] — in device
/// space, with its clip already worked out — to a flat list, and that list is replayed into a
/// [`Canvas`] afterwards, one damage rectangle at a time. The split is the design: the recording
/// is what the damage between two frames is computed from, and the replay is where the clip pays
/// off, because the rasteriser takes it into the scan rather than rejecting pixels at blend time.
///
/// The list is flat on purpose. iced keeps a tree of layers, and `iced_tiny_skia` compares those
/// layers item by item with an all-or-nothing rule (`Item::Group` never compares equal), so one
/// canvas that redraws a stroke animation damages everything it covers, every frame. Here the
/// commands inside a layer are ordinary items, so the ones that did not change can be recognised
/// as unchanged and left alone.
pub struct Renderer {
    default_font: Font,
    default_size: Pixels,
    /// What has been recorded since the last reset, in the order it was drawn.
    items: Vec<Item>,
    /// The clip stack: one rectangle per open layer, already in device space.
    layers: Vec<Rectangle>,
    /// The transformation stack. See `start_transformation`.
    transformations: Vec<Transformation>,
    /// The bounds the frame was reset to, which is the clip before any layer opens.
    bounds: Rectangle,
    /// The shaped paragraphs, by what they say. iced's own cache, the one its backends share.
    paragraphs: RefCell<iced_graphics::text::cache::Cache>,
    /// The rasterised glyph masks.
    glyphs: RefCell<crate::text::Glyphs>,
}

impl Renderer {
    /// Creates a renderer with a default font and size.
    pub fn new(default_font: Font, default_size: Pixels) -> Self {
        Self {
            default_font,
            default_size,
            items: Vec::new(),
            layers: Vec::new(),
            transformations: Vec::new(),
            bounds: Rectangle::with_size(Size::ZERO),
            paragraphs: RefCell::new(iced_graphics::text::cache::Cache::new()),
            glyphs: RefCell::new(crate::text::Glyphs::new()),
        }
    }

    /// What has been recorded since the last reset, in the order it was drawn.
    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// How many transformations are open.
    pub fn open_transformations(&self) -> usize {
        self.transformations.len()
    }

    /// The translation and scale in force, folded down from every open transformation.
    ///
    /// Transformations compose, so two of them are not two placements: the inner one's translation
    /// is scaled by the outer one before it is added.
    fn placement(&self) -> Placement {
        self.transformations
            .iter()
            .fold(Placement::IDENTITY, |outer, inner| Placement {
                translation: outer.translation + inner.translation() * outer.scale,
                scale: outer.scale * inner.scale_factor(),
            })
    }

    /// Replays the recording into `canvas`, clipped to `damage`.
    pub fn replay(&self, canvas: &mut Canvas<'_>, damage: GfxRect) {
        for item in &self.items {
            match &item.primitive {
                // Text is the one command that cannot be replayed by a free function: drawing it
                // needs the shaping and glyph caches, and they live here.
                Primitive::Text {
                    position,
                    color,
                    run,
                    ..
                } => self.draw_text(
                    canvas,
                    *position,
                    *color,
                    run,
                    item.clip,
                    damage,
                    item.placement,
                ),
                primitive => geometry::draw(canvas, primitive, item.clip, damage, item.placement),
            }
        }
    }

    /// Draws one recorded run of text, clipped to `damage`.
    #[allow(clippy::too_many_arguments)]
    fn draw_text(
        &self,
        canvas: &mut Canvas<'_>,
        position: Point,
        color: Color,
        run: &TextRun,
        clip_bounds: Rectangle,
        damage: GfxRect,
        placement: Placement,
    ) {
        let Some(clip) = rect(clip_bounds.intersection(&Rectangle {
            x: damage.left(),
            y: damage.top(),
            width: damage.width,
            height: damage.height,
        })) else {
            return;
        };

        let font_system = iced_graphics::text::font_system();
        let mut font_system = font_system.write().expect("the font system");

        canvas.save();
        canvas.clip_rect(clip);
        geometry::place(canvas, placement);

        match run {
            TextRun::Parameters(parameters) => {
                let mut paragraphs = self.paragraphs.borrow_mut();
                let (_, entry) = paragraphs.allocate(font_system.raw(), parameters.key());

                self.glyphs.borrow_mut().draw(
                    canvas,
                    font_system.raw(),
                    &entry.buffer,
                    position,
                    color,
                );
            }
            // The paragraph is shaped and owned by whoever drew it; a weak reference that no
            // longer upgrades means it went away between the recording and the replay, which is
            // nothing to draw rather than an error.
            TextRun::Shaped(paragraph) => {
                if let Some(paragraph) = paragraph.upgrade() {
                    self.glyphs.borrow_mut().draw(
                        canvas,
                        font_system.raw(),
                        paragraph.buffer(),
                        position,
                        color,
                    );
                }
            }
        }

        canvas.restore();
    }

    /// The clip in force: the frame's bounds, narrowed by every open layer.
    fn clip(&self) -> Rectangle {
        self.layers.iter().fold(self.bounds, |clip, bounds| {
            clip.intersection(bounds)
                .unwrap_or(Rectangle::with_size(Size::ZERO))
        })
    }

    /// Records one command under the clip in force.
    fn record(&mut self, primitive: Primitive) {
        let clip = self.clip();

        self.record_within(primitive, clip);
    }

    /// Records one command under the clip in force, narrowed by one the caller asked for.
    ///
    /// A widget can hand a text command a smaller rectangle than its layer (a label that must not
    /// spill out of its own box), and narrowing here is what keeps the recording flat: the clip
    /// travels with the command rather than becoming another layer.
    fn record_within(&mut self, primitive: Primitive, clip: Rectangle) {
        let clip = self
            .clip()
            .intersection(&clip)
            .unwrap_or(Rectangle::with_size(Size::ZERO));
        let placement = self.placement();

        self.items.push(Item {
            primitive,
            clip,
            placement,
        });
    }
}

// The bodies are deliberately bare: the associated types and constants above are the ones that
// cannot be inferred, and compiling this is how the methods got enumerated — the compiler lists
// every missing item with its exact signature, which is a better source than any document. Each
// `todo!()` below is named after what it will do.

impl iced_core::renderer::Renderer for Renderer {
    /// Opens a layer, clipped to `bounds`.
    fn start_layer(&mut self, bounds: Rectangle) {
        self.layers.push(self.placement().map(bounds));
    }

    /// Closes the current layer.
    fn end_layer(&mut self) {
        self.layers.pop();
    }

    /// Pushes a transformation.
    ///
    /// Not a no-op, and the first draft of this was wrong about that: nothing calls
    /// `start_transformation` *directly*, but `with_translation` and `with_transformation` are
    /// provided methods on the trait that call it, and iced's `scrollable` scrolls its content with
    /// exactly that. Missing it looked like this: the scrollbar moved (its own arithmetic) while the
    /// content stayed where it was.
    ///
    /// What is kept is the flat part — a translation and a scale, which is what `Transformation`
    /// exposes and what iced's own backends take from it.
    fn start_transformation(&mut self, transformation: Transformation) {
        self.transformations.push(transformation);
    }

    /// Pops the last transformation.
    fn end_transformation(&mut self) {
        self.transformations.pop();
    }

    /// Records a quad: a rounded rectangle, a border, a shadow, optionally snapped to the grid.
    ///
    /// The fill and the border are two commands, because that is what they are: a fill inside the
    /// bounds and a line inside its edge. A solid fill reaches `Canvas::draw_rrect`, which is the
    /// fast axis-aligned path with corner arcs; a gradient reaches the shader-aware quad filler.
    fn fill_quad(&mut self, quad: Quad, background: impl Into<Background>) {
        let rrect = rrect(&quad);

        // The shadow is not drawn: a blurred one is a per-pixel distance-field loop, and no widget
        // in this OS asks for one. `iced_tiny_skia` skips it with a warning for the same reason.
        let _ = quad.shadow;

        self.record(Primitive::Rounded {
            rrect,
            paint: paint(background.into(), quad.bounds),
        });

        if quad.border.width > 0.0 {
            self.record(Primitive::RoundedStroke {
                rrect,
                paint: geometry::solid_paint(quad.border.color),
                width: quad.border.width,
            });
        }
    }

    /// Clears everything and starts again from `bounds`.
    fn reset(&mut self, bounds: Rectangle) {
        self.bounds = bounds;
        self.items.clear();
        self.layers.clear();
        self.transformations.clear();
    }

    /// Reports that an image cannot be allocated.
    ///
    /// Images need `iced_graphics`' `image` feature and the `image` crate in the firmware. The
    /// launcher has none on purpose — its wallpaper and icons are baked RGB565 blits that
    /// `pomelo-gfx` draws natively — so this is `Unsupported` rather than missing.
    fn allocate_image(
        &mut self,
        handle: &iced_core::image::Handle,
        callback: impl FnOnce(Result<image::Allocation, image::Error>) + Send + 'static,
    ) {
        let _ = handle;
        callback(Err(image::Error::Unsupported));
    }
}

impl iced_geometry::Renderer for Renderer {
    type Geometry = geometry::Geometry;
    type Frame = geometry::Frame;

    /// A frame that records into `bounds`.
    fn new_frame(&self, bounds: Rectangle) -> Self::Frame {
        geometry::Frame::new(bounds)
    }

    /// Records a canvas's geometry.
    ///
    /// The clip is whichever is narrower, this frame's or the geometry's own, and every command
    /// the canvas drew becomes an item of its own. That last part is the point: a canvas that
    /// redraws a stroke animation appends the strokes that are already finished as *equal* items,
    /// so the damage is the stroke that is still growing rather than the whole picture.
    ///
    /// The geometry's clip is in the canvas's *own* coordinates -- a canvas states its frame as
    /// its own size, and the widget wraps the draw in a translation -- so it is placed before the
    /// comparison. A canvas that is not at the origin is the case that needs it: 480x436 at y=44
    /// clips its own frame to 0..436, and intersecting that with the screen would cut the last 44
    /// rows off the bottom of every command.
    fn draw_geometry(&mut self, geometry: Self::Geometry) {
        let clip = self
            .clip()
            .intersection(&self.placement().map(geometry.clip_bounds()))
            .unwrap_or(Rectangle::with_size(Size::ZERO));

        for primitive in geometry.primitives() {
            let placement = self.placement();

            self.items.push(Item {
                primitive: primitive.clone(),
                clip,
                placement,
            });
        }
    }
}

impl text::Renderer for Renderer {
    type Font = Font;
    type Paragraph = iced_graphics::text::Paragraph;
    type Editor = iced_graphics::text::Editor;

    const ICON_FONT: Font = Font::with_name("Iced-Icons");
    const CHECKMARK_ICON: char = '\u{f00c}';
    const ARROW_DOWN_ICON: char = '\u{e800}';
    const ICED_LOGO: char = '\u{e801}';
    const SCROLL_UP_ICON: char = '\u{e802}';
    const SCROLL_DOWN_ICON: char = '\u{e803}';
    const SCROLL_LEFT_ICON: char = '\u{e804}';
    const SCROLL_RIGHT_ICON: char = '\u{e805}';

    fn default_font(&self) -> Self::Font {
        self.default_font
    }

    fn default_size(&self) -> Pixels {
        self.default_size
    }

    /// Records a laid-out paragraph.
    ///
    /// The paragraph is the shared `iced_graphics` one, so this is also where the glyphs come
    /// from: each is a coverage mask, which `Canvas::blit_mask` blends in a colour with a smooth
    /// edge. No antialiasing is ours to write.
    /// Records a paragraph that has already been shaped.
    ///
    /// The `Text` widget shapes its own, in its own tree state, and re-shapes it only when the text
    /// changes — so this takes a weak reference to it rather than shaping anything of its own.
    fn fill_paragraph(
        &mut self,
        paragraph: &Self::Paragraph,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let paragraph = paragraph.downgrade();

        self.record_within(
            Primitive::Text {
                position,
                color,
                // How big the paragraph says it is, which is the best anyone can know: `Paragraph`
                // exposes its buffer and this. It is also what iced's own backends report for one —
                // using the widget's clip instead reports a label as big as the screen it sits on,
                // which is exactly what the first draft of this did.
                bounds: Rectangle::new(position, paragraph.min_bounds),
                run: TextRun::Shaped(paragraph),
            },
            clip_bounds,
        );
    }

    /// Records an editor, which is a paragraph that is being typed into.
    fn fill_editor(
        &mut self,
        editor: &Self::Editor,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        let _ = (editor, position, color, clip_bounds);
        todo!("record the editor's glyphs")
    }

    /// Records a run of text. iced hands this the raw text and its geometry, so it is shaped when
    /// it is drawn — through the shared cache, or every label would be re-shaped every frame.
    fn fill_text(
        &mut self,
        text: text::Text<String, Self::Font>,
        position: Point,
        color: Color,
        clip_bounds: Rectangle,
    ) {
        self.record_within(
            Primitive::Text {
                position,
                color,
                // The text's own box, which is what iced's backends report for one run of text: the
                // clip a widget passes is usually its whole area, and using that would report a
                // label as big as the screen it sits on.
                bounds: Rectangle::new(position, text.bounds),
                run: TextRun::Parameters(Parameters {
                    content: text.content,
                    size: text.size.0,
                    line_height: text.line_height.to_absolute(text.size).0,
                    font: text.font,
                    align_x: text.align_x,
                    shaping: text.shaping,
                    bounds: text.bounds,
                }),
            },
            clip_bounds,
        );
    }
}

impl mesh::Renderer for Renderer {
    /// Ignores a mesh. `tiny-skia`'s backend does the same outside its `geometry` feature, and
    /// neither the launcher nor the calculator draws one.
    fn draw_mesh(&mut self, mesh: mesh::Mesh) {
        let _ = mesh;
    }

    fn draw_mesh_cache(&mut self, cache: mesh::Cache) {
        let _ = cache;
    }
}

/// The compositor this renderer reports, which is none.
///
/// `iced_program::Renderer` — the bound a `Program` states — asks for `compositor::Default`,
/// because on a desktop the compositor is what creates a window's surface and presents into it.
/// There is no window here: `iced_winit` owns the panel's buffer and presents the damaged
/// regions itself, so nothing in this stack ever constructs one of these. The type exists to
/// satisfy the bound, and the methods that would touch a surface say what they cannot do instead
/// of pretending to do it.
#[derive(Debug)]
pub struct NoSurface;

impl iced_graphics::compositor::Compositor for NoSurface {
    type Renderer = Renderer;
    type Surface = ();

    async fn with_backend(
        _settings: iced_graphics::Settings,
        _display: impl iced_graphics::compositor::Display + Clone,
        _compatible_window: impl iced_graphics::compositor::Window + Clone,
        _shell: iced_graphics::Shell,
        _backend: Option<&str>,
    ) -> Result<Self, iced_graphics::Error> {
        Ok(Self)
    }

    fn create_renderer(&self) -> Renderer {
        Renderer::new(Font::default(), Pixels(16.0))
    }

    fn create_surface<W: iced_graphics::compositor::Window + Clone>(
        &mut self,
        _window: W,
        _width: u32,
        _height: u32,
    ) {
    }

    fn configure_surface(&mut self, _surface: &mut (), _width: u32, _height: u32) {}

    fn load_font(&mut self, font: Cow<'static, [u8]>) {
        iced_graphics::text::font_system()
            .write()
            .expect("the font system")
            .load_font(font);
    }

    fn information(&self) -> iced_graphics::compositor::Information {
        iced_graphics::compositor::Information {
            adapter: String::from("pomelo-gfx"),
            backend: String::from("RGB565"),
        }
    }

    fn present(
        &mut self,
        _renderer: &mut Renderer,
        _surface: &mut (),
        _viewport: &iced_graphics::Viewport,
        _background: Color,
        _on_pre_present: impl FnOnce(),
    ) -> Result<(), iced_graphics::compositor::SurfaceError> {
        panic!("`pomelo-gfx` has no window surface: `iced_winit` presents the panel")
    }

    fn screenshot(
        &mut self,
        renderer: &mut Renderer,
        viewport: &iced_graphics::Viewport,
        background: Color,
    ) -> Vec<u8> {
        iced_core::renderer::Headless::screenshot(
            renderer,
            viewport.physical_size(),
            viewport.scale_factor(),
            background,
        )
    }
}

impl iced_graphics::compositor::Default for Renderer {
    type Compositor = NoSurface;
}

impl iced_core::renderer::Headless for Renderer {
    async fn new(
        default_font: Font,
        default_text_size: Pixels,
        backend: Option<&str>,
    ) -> Option<Self> {
        // `None` is "whatever the platform draws with". A named backend that is not this one is a
        // request that cannot be honoured, and saying so is better than drawing with the wrong
        // rasteriser.
        matches!(backend, None | Some("pomelo-gfx" | "pomelo_gfx"))
            .then(|| Renderer::new(default_font, default_text_size))
    }

    fn name(&self) -> String {
        String::from("pomelo-gfx")
    }

    /// Draws the recording into a fresh RGB565 buffer and hands it back as RGBA8888.
    ///
    /// The one place this crate makes pixels of its own: it needs no surface and no window, which
    /// is why it can be the real thing rather than a stub. iced's testing helpers are what ask.
    fn screenshot(&mut self, size: Size<u32>, _scale_factor: f32, background: Color) -> Vec<u8> {
        let Some(mut panel) = Pixmap565::new(size.width, size.height) else {
            return Vec::new();
        };

        let damage = GfxRect::from_ltrb(0.0, 0.0, size.width as f32, size.height as f32);
        let mut canvas = Canvas::new(panel.as_mut());

        canvas.clear(geometry::color_of(background));
        self.replay(&mut canvas, damage);

        panel
            .data()
            .iter()
            .flat_map(|pixel| {
                let (red, green, blue) = (pixel >> 11, (pixel >> 5) & 0x3f, pixel & 0x1f);

                [
                    (red << 3 | red >> 2) as u8,
                    (green << 2 | green >> 4) as u8,
                    (blue << 3 | blue >> 2) as u8,
                    0xff,
                ]
            })
            .collect()
    }
}

/// A `Rectangle` as the rasteriser states it, if it has any area at all.
fn rect(bounds: Option<Rectangle>) -> Option<GfxRect> {
    let bounds = bounds?;

    (bounds.width > 0.0 && bounds.height > 0.0).then(|| {
        GfxRect::from_ltrb(
            bounds.x,
            bounds.y,
            bounds.x + bounds.width,
            bounds.y + bounds.height,
        )
    })
}

/// A quad's rectangle and corner radius, as the rasteriser states them.
fn rrect(quad: &Quad) -> RRect {
    let bounds = quad.bounds;
    let radius = quad.border.radius;

    // The rasteriser has one radius per rectangle, and iced has one per corner. Equal corners —
    // which is what every widget in this OS asks for — convert exactly. When they differ the
    // roundest one is used, so a corner is never sharper than it was asked to be; a genuinely
    // mixed-radius rectangle would need the rasteriser to carry four corner arcs.
    let radius = radius
        .top_left
        .max(radius.top_right)
        .max(radius.bottom_right)
        .max(radius.bottom_left);

    RRect::from_rect_radius(
        GfxRect::from_ltrb(
            bounds.x,
            bounds.y,
            bounds.x + bounds.width,
            bounds.y + bounds.height,
        ),
        Radius::circular(radius),
    )
}

/// A background as a paint, in the quad's own rectangle.
///
/// The rectangle is not decoration: iced states a widget's gradient as an *angle*, and an angle
/// only becomes two endpoints once there is a shape to measure it against.
fn paint(background: Background, bounds: Rectangle) -> pomelo_gfx::Paint<'static> {
    match background {
        Background::Color(color) => geometry::solid_paint(color),
        Background::Gradient(gradient) => geometry::gradient_paint(gradient, bounds),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_core::border::Radius as IcedRadius;
    use iced_core::gradient::Linear;
    use iced_core::renderer::Renderer as _;
    use iced_core::{Border, Color as IcedColor, Size as IcedSize};
    use iced_graphics::geometry::frame::Backend as _;
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
        renderer.replay(&mut canvas, damage);
        drop(canvas);

        pixmap
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

        assert_eq!(renderer.items().len(), 1, "one quad is one command");

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
            renderer.items().len(),
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
    fn a_canvas_geometry_becomes_one_item_per_command() {
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

        assert_eq!(
            renderer.items().len(),
            2,
            "each command the canvas drew is an item of its own"
        );

        let pixmap = render(&renderer);

        assert_eq!(at(&pixmap, 8, 8), (255, 255, 255));
        assert_eq!(at(&pixmap, 8, 40), (255, 255, 255));
        assert_eq!(at(&pixmap, 8, 24), (0, 0, 0), "and nothing between them");
    }

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
            renderer.items()[0].clip,
            layer,
            "the layer is the clip the command was recorded under"
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

        let bounds = quad.bounds;
        renderer.fill_quad(quad, Background::Color(IcedColor::WHITE));

        assert_eq!(
            renderer.items()[0].bounds(),
            bounds,
            "a command's damage is the part of it inside its clip"
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

        assert_eq!(
            renderer.items()[0].bounds(),
            Rectangle::new(Point::new(0.0, offset), size),
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

        assert_eq!(renderer.items().len(), 0, "the recording is empty again");

        renderer.fill_quad(white_quad(), Background::Color(IcedColor::WHITE));

        assert_eq!(
            renderer.items()[0].clip,
            Rectangle::with_size(IcedSize::new(SIZE as f32, SIZE as f32)),
            "and the layer that was open is gone too"
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
}
