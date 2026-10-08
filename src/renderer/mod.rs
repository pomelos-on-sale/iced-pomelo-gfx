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

use std::cell::RefCell;

use iced_core::image;
use iced_core::renderer::Quad;
use iced_core::text;
use iced_core::{Background, Color, Font, Pixels, Point, Rectangle, Size, Transformation, Vector};
use iced_graphics::geometry as iced_geometry;
use iced_graphics::layer::Stack;
use iced_graphics::mesh;
use pomelo_gfx::{Canvas, Pixmap565, RRect, Radius, Rect as GfxRect};

use crate::geometry::{self, Parameters, Primitive, TextRun};
use crate::layer::{Item, Layer, Text};

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

    /// The placement a transformation comes down to.
    ///
    /// `Stack` composes transformations as the `Mat4` products they are, so this is where the
    /// composition is undone into the two numbers the rasteriser takes.
    pub fn of(transformation: Transformation) -> Self {
        Self {
            translation: transformation.translation(),
            scale: transformation.scale_factor(),
        }
    }

    /// A rectangle in this placement's coordinates, in the destination's.
    pub fn map(&self, bounds: Rectangle) -> Rectangle {
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

/// The renderer iced drives.
///
/// It **records** rather than draws. Every call iced makes appends a command to the current
/// [`Layer`] — the shape `iced_tiny_skia` records into, held in iced's own
/// [`Stack`](iced_graphics::layer::Stack) — and the recording is replayed into a [`Canvas`]
/// afterwards, one damage rectangle at a time. The split is the design: the recording is what the
/// damage between two frames is computed from ([`Layer::damage`]), and the replay is where the clip
/// pays off, because the rasteriser takes it into the scan rather than rejecting pixels at blend
/// time.
pub struct Renderer {
    default_font: Font,
    default_size: Pixels,
    /// The frame being recorded, and the clip and transformation stack around it.
    layers: Stack<Layer>,
    /// How many transformations are open. `Stack` does not report its own depth, and this is what
    /// the diagnostics ask for.
    transformations: usize,
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
            layers: Stack::new(),
            transformations: 0,
            paragraphs: RefCell::new(iced_graphics::text::cache::Cache::new()),
            glyphs: RefCell::new(crate::text::Glyphs::new(crate::baked::get())),
        }
    }

    /// The layers that were recorded since the last reset, in the order they were drawn.
    pub fn layers(&mut self) -> &[Layer] {
        self.layers.flush();

        self.layers.as_slice()
    }

    /// How many transformations are open.
    pub fn open_transformations(&self) -> usize {
        self.transformations
    }

    /// Blits a baked 16-bit RGB565 bitmap image with an 8-bit alpha mask into the current layer.
    pub fn draw_bitmap_565(
        &mut self,
        bounds: Rectangle,
        width: u16,
        height: u16,
        rgb565: &'static [u16],
        alpha: &'static [u8],
    ) {
        let (layer, transformation) = self.layers.current_mut();
        let placement = Placement::of(transformation);
        let mapped = placement.map(bounds);
        layer.quads.push(Primitive::Image565 {
            rect: GfxRect::from_ltwh(mapped.x, mapped.y, mapped.width, mapped.height),
            pixels: rgb565,
            alpha: Some(alpha),
            src_w: width as u32,
            src_h: height as u32,
        });
    }

    /// Draws the recording into `canvas`, clipped to `damage`.
    ///
    /// The layer's bounds are the clip for the commands that have none of their own, and a canvas's
    /// commands are clipped to what the canvas said, placed, and then to the layer.
    pub fn draw(&self, canvas: &mut Canvas<'_>, damage: GfxRect) {
        for layer in self.layers.as_slice() {
            let quads = crate::profile::start(crate::profile::Phase::Quads);

            for command in &layer.quads {
                geometry::draw(canvas, command, layer.bounds, damage, Placement::IDENTITY);
            }

            drop(quads);

            for item in &layer.primitives {
                let placement = Placement::of(item.transformation());

                let Some(clip) = placement
                    .map(item.clip_bounds())
                    .intersection(&layer.bounds)
                else {
                    continue;
                };

                for command in item.as_slice() {
                    geometry::draw(canvas, command, clip, damage, placement);
                }
            }

            let _text = crate::profile::start(crate::profile::Phase::Text);

            for item in &layer.text {
                for text in item.as_slice() {
                    self.draw_text(canvas, text, damage);
                }
            }
        }
    }

    /// Draws one recorded run of text, clipped to `damage`.
    fn draw_text(&self, canvas: &mut Canvas<'_>, text: &Text, damage: GfxRect) {
        let Some(clip) = rect(text.clip.intersection(&Rectangle {
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
        geometry::place(canvas, Placement::of(text.transformation));

        match &text.run {
            TextRun::Parameters(parameters) => {
                let mut paragraphs = self.paragraphs.borrow_mut();
                let (_, entry) = paragraphs.allocate(font_system.raw(), parameters.key());

                self.glyphs.borrow_mut().draw(
                    canvas,
                    font_system.raw(),
                    &entry.buffer,
                    text.position,
                    text.color,
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
                        text.position,
                        text.color,
                    );
                }
            }
        }

        canvas.restore();
    }

    /// Records a run of text into the current layer.
    ///
    /// The text's own box and its clip are placed as they are recorded, which is what
    /// `iced_tiny_skia` does with its own text commands, and the transformation travels with the
    /// run because that is what places the glyphs when the frame is drawn.
    fn push_text(
        &mut self,
        position: Point,
        color: Color,
        bounds: Rectangle,
        run: TextRun,
        clip_bounds: Rectangle,
    ) {
        let (layer, transformation) = self.layers.current_mut();
        let placement = Placement::of(transformation);

        let Some(clip) = placement.map(clip_bounds).intersection(&layer.bounds) else {
            return;
        };

        layer.text.push(Item::Live(Text {
            position,
            color,
            bounds: placement.map(bounds),
            clip,
            run,
            transformation,
        }));
    }
}

// The bodies are deliberately bare: the associated types and constants above are the ones that
// cannot be inferred, and compiling this is how the methods got enumerated — the compiler lists
// every missing item with its exact signature, which is a better source than any document. Each
// `todo!()` below is named after what it will do.

impl iced_core::renderer::Renderer for Renderer {
    /// Opens a layer, clipped to `bounds`.
    ///
    /// iced's own stack does the work: the bounds are placed through the transformation in force
    /// and become the clip every command in the layer is recorded inside.
    fn start_layer(&mut self, bounds: Rectangle) {
        self.layers.push_clip(bounds);
    }

    /// Closes the current layer.
    fn end_layer(&mut self) {
        self.layers.pop_clip();
    }

    /// Pushes a transformation.
    ///
    /// Not a no-op, and the first draft of this was wrong about that: nothing calls
    /// `start_transformation` *directly*, but `with_translation` and `with_transformation` are
    /// provided methods on the trait that call it, and iced's `scrollable` scrolls its content with
    /// exactly that. Missing it looked like this: the scrollbar moved (its own arithmetic) while the
    /// content stayed where it was.
    ///
    /// The stack composes the matrices; the flat part is taken back out when a command is recorded
    /// or drawn (see [`Placement`]).
    fn start_transformation(&mut self, transformation: Transformation) {
        self.transformations += 1;
        self.layers.push_transformation(transformation);
    }

    /// Pops the last transformation.
    fn end_transformation(&mut self) {
        self.transformations -= 1;
        self.layers.pop_transformation();
    }

    /// Records a quad: a rounded rectangle, a border, a shadow, optionally snapped to the grid.
    ///
    /// The fill and the border are two commands, because that is what they are: a fill inside the
    /// bounds and a line inside its edge. A solid fill reaches `Canvas::draw_rrect`, which is the
    /// fast axis-aligned path with corner arcs; a gradient reaches the shader-aware quad filler.
    ///
    /// The transformation in force is placed into the commands as they are recorded, because the
    /// rasteriser takes a transform rather than a `Quad`: the rectangle, its radius, its border
    /// width and the rectangle a gradient is measured against all move together.
    fn fill_quad(&mut self, quad: Quad, background: impl Into<Background>) {
        let (layer, transformation) = self.layers.current_mut();
        let placement = Placement::of(transformation);
        let bounds = placement.map(quad.bounds);
        let rrect = rrect(&quad, placement);

        // The shadow is not drawn: a blurred one is a per-pixel distance-field loop, and no widget
        // in this OS asks for one. `iced_tiny_skia` skips it with a warning for the same reason.
        let _ = quad.shadow;

        layer.quads.push(Primitive::Rounded {
            rrect,
            paint: paint(background.into(), bounds),
        });

        if quad.border.width > 0.0 {
            layer.quads.push(Primitive::RoundedStroke {
                rrect,
                paint: geometry::solid_paint(quad.border.color),
                width: quad.border.width * placement.scale,
            });
        }
    }

    /// Clears everything and starts again from `bounds`.
    fn reset(&mut self, bounds: Rectangle) {
        self.layers.reset(bounds);
        self.transformations = 0;
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

    /// Records a canvas's geometry as one item.
    ///
    /// The item is the shape `iced_tiny_skia` records too: the canvas's own clip and the
    /// transformation in force travel with it, and [`Layer::damage`] is what finds the commands
    /// inside. A `canvas::Cache` arrives as an `Arc` that is compared by identity, so a canvas that
    /// caches its static part costs one pointer comparison per frame — and one that does not still
    /// only damages the piece that changed, because a group that differs is walked command by
    /// command.
    ///
    /// The geometry's clip is in the canvas's *own* coordinates -- a canvas states its frame as
    /// its own size, and the widget wraps the draw in a translation -- so it is placed by whoever
    /// draws or compares it. A canvas that is not at the origin is the case that needs it: 480x436
    /// at y=44 clips its own frame to 0..436, and intersecting that with the screen before placing
    /// it would cut the last 44 rows off the bottom of every command.
    fn draw_geometry(&mut self, geometry: Self::Geometry) {
        let (layer, transformation) = self.layers.current_mut();

        let item = match geometry {
            crate::geometry::Geometry::Live {
                primitives,
                clip_bounds,
            } => Item::Group(primitives, clip_bounds, transformation),
            crate::geometry::Geometry::Cache(cache) => {
                Item::Cached(cache.primitives, cache.clip_bounds, transformation)
            }
        };

        layer.primitives.push(item);
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

        // How big the paragraph says it is, which is the best anyone can know: `Paragraph`
        // exposes its buffer and this. It is also what iced's own backends report for one —
        // using the widget's clip instead reports a label as big as the screen it sits on,
        // which is exactly what the first draft of this did.
        self.push_text(
            position,
            color,
            Rectangle::new(position, paragraph.min_bounds),
            TextRun::Shaped(paragraph),
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
        // The text's own box, which is what iced's backends report for one run of text: the clip a
        // widget passes is usually its whole area, and using that would report a label as big as
        // the screen it sits on.
        self.push_text(
            position,
            color,
            Rectangle::new(position, text.bounds),
            TextRun::Parameters(Parameters {
                content: text.content,
                size: text.size.0,
                line_height: text.line_height.to_absolute(text.size).0,
                font: text.font,
                align_x: text.align_x,
                shaping: text.shaping,
                bounds: text.bounds,
            }),
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
        self.draw(&mut canvas, damage);

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

/// A quad's rectangle and corner radius, as the rasteriser states them, placed.
fn rrect(quad: &Quad, placement: Placement) -> RRect {
    let bounds = placement.map(quad.bounds);
    let radius = quad.border.radius;

    // The rasteriser has one radius per rectangle, and iced has one per corner. Equal corners —
    // which is what every widget in this OS asks for — convert exactly. When they differ the
    // roundest one is used, so a corner is never sharper than it was asked to be; a genuinely
    // mixed-radius rectangle would need the rasteriser to carry four corner arcs.
    let radius = radius
        .top_left
        .max(radius.top_right)
        .max(radius.bottom_right)
        .max(radius.bottom_left)
        * placement.scale;

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

/// A background as a paint, in the rectangle it was measured against.
///
/// The rectangle is not decoration: iced states a widget's gradient as an *angle*, and an angle
/// only becomes two endpoints once there is a shape to measure it against. It is the *placed*
/// rectangle, because the command it paints is placed too.
fn paint(background: Background, bounds: Rectangle) -> pomelo_gfx::Paint<'static> {
    match background {
        Background::Color(color) => geometry::solid_paint(color),
        Background::Gradient(gradient) => geometry::gradient_paint(gradient, bounds),
    }
}


#[cfg(test)]
mod tests;
