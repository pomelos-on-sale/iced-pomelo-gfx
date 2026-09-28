//! What drawing text costs, in time rather than bytes.
//!
//! The mechanism first, because the question "is it vector, and is that not recomputed every
//! frame?" has a yes and a no in it:
//!
//! 1. **Shaping** — `Paragraph::with_text` hands the string to `cosmic-text`, which turns it into
//!    glyph runs with positions (font tables, kerning, bidi, line breaking). Once per paragraph,
//!    and only when the text changes: the `Text` widget keeps the paragraph in its own tree state.
//! 2. **Rasterising** — a glyph is an outline until something needs its pixels. The first time a
//!    `(glyph, size)` is asked for, `swash` walks the outline and produces a coverage byte per
//!    pixel; our renderer keeps that in `Glyph::masks` for the life of the renderer.
//! 3. **Blitting** — every frame after that, the cached mask is blended into the RGB565 frame
//!    buffer. This is the only per-frame cost, and it is a memory loop, not a vector one.
//!
//! So the vector work happens once per `(glyph, size)`, not once per frame — and the numbers below
//! are what "once" costs. They are measured on the *host*, which is 20-30× faster than the LX7 at
//! this kind of scalar work; the section at the bottom of the output says how to read them for the
//! board. A CJK font can be measured too, by pointing `POMELO_MEASURE_FONT` at one:
//!
//! ```text
//! POMELO_MEASURE_FONT=/tmp/cjk-3755.ttf \
//!     cargo test -p iced-pomelo-gfx --test text_cost_tests -- --nocapture
//! ```

use std::borrow::Cow;
use std::time::{Duration, Instant};

use iced_core::renderer::Renderer as _;
use iced_core::text::Renderer as _;
use iced_core::text::{LineHeight, Shaping, Wrapping};
use iced_core::{Font, Pixels, Point, Rectangle, Size, Text};
use pomelo_gfx::Canvas;

use iced_pomelo_gfx::Renderer;

/// The Latin subset the OS embeds: 16 KiB, 99 glyphs.
const SUBSET: &[u8] = include_bytes!("../../../assets/fonts/source/Roboto-Subset.ttf");

/// Printable ASCII: the glyph set a screen of this OS's text is drawn from.
const ASCII: &str = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

const PANEL: u32 = 480;

/// Runs `body` until it has taken at least `MIN_ROUNDS`, and returns the *fastest* round.
///
/// The fastest and not the average: a shared machine's scheduler can only make a round slower, so
/// the minimum is the closest thing to the cost of the work itself.
fn fastest<F: FnMut()>(rounds: usize, mut body: F) -> Duration {
    let mut best = Duration::MAX;

    for _ in 0..rounds {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }

    best
}

fn micros(duration: Duration, per: usize) -> f64 {
    duration.as_secs_f64() * 1e6 / per as f64
}

/// Installs `bytes` as iced's default font.
///
/// The same two steps the host crate's `fonts::install` takes — `load_font`, and then make the face
/// that was just added *the* sans-serif family, because `Font::default()` is `Family::SansSerif`
/// and iced's only default face would otherwise be its icon font.
fn install(bytes: &'static [u8]) {
    let mut system = iced_graphics::text::font_system()
        .write()
        .expect("write font system");

    system.load_font(Cow::Borrowed(bytes));

    let family = system
        .raw()
        .db_mut()
        .faces()
        .last()
        .and_then(|face| face.families.first())
        .map(|(family, _)| family.clone());

    if let Some(family) = family {
        system.raw().db_mut().set_sans_serif_family(family);
    }
}

/// Shapes `text` at `size`, and draws it into a fresh 480×480 panel.
///
/// Returns the three phases, separated the only way they can be from outside the crate: the
/// shaping happens when the paragraph is built, and the rasterising on the first replay into a
/// renderer that has never seen these glyphs. A second replay into the *same* renderer is the
/// blit alone.
fn phases(font_bytes: &'static [u8], text: &str, size: f32) -> (Duration, Duration, Duration) {
    install(font_bytes);

    let shape = fastest(3, || {
        let _ = paragraph(text, size);
    });

    let bounds = Rectangle::with_size(Size::new(PANEL as f32, PANEL as f32));
    let clip = bounds;
    let paragraph = paragraph(text, size);
    let mut panel = pomelo_gfx::Pixmap565::new(PANEL, PANEL).expect("a panel");

    // Cold: a *fresh renderer* every round, so every glyph is rasterised again. This is the first
    // frame that shows this text — and the reason the masks below are worth having.
    let cold = fastest(3, || {
        let mut renderer = Renderer::new(Font::default(), Pixels(size));

        renderer.reset(bounds);
        renderer.fill_paragraph(&paragraph, Point::ORIGIN, iced_core::Color::WHITE, clip);
        replay(&renderer, &mut panel);
    });

    // Warm: one renderer that has seen every glyph, replayed again — the per-frame cost.
    let mut renderer = Renderer::new(Font::default(), Pixels(size));

    renderer.reset(bounds);
    renderer.fill_paragraph(&paragraph, Point::ORIGIN, iced_core::Color::WHITE, clip);
    replay(&renderer, &mut panel);

    let warm = fastest(3, || replay(&renderer, &mut panel));

    (shape, cold, warm)
}

fn replay(renderer: &Renderer, panel: &mut pomelo_gfx::Pixmap565) {
    let damage = pomelo_gfx::Rect::from_ltrb(0.0, 0.0, PANEL as f32, PANEL as f32);
    let mut canvas = Canvas::new(panel.as_mut());

    renderer.replay(&mut canvas, damage);
}

fn paragraph(text: &str, size: f32) -> iced_graphics::text::Paragraph {
    <iced_graphics::text::Paragraph as iced_core::text::Paragraph>::with_text(Text {
        content: text,
        bounds: Size::new(PANEL as f32, PANEL as f32),
        size: Pixels(size),
        line_height: LineHeight::Relative(1.2),
        font: Font::default(),
        align_x: iced_core::alignment::Horizontal::Left.into(),
        align_y: iced_core::alignment::Vertical::Top,
        shaping: Shaping::Basic,
        wrapping: Wrapping::default(),
    })
}

#[test]
fn what_text_costs_in_time() {
    println!("\n=== drawing text, measured (host: 20-30x faster than the LX7) ===\n");

    for size in [15.0_f32, 22.0, 38.0, 96.0] {
        let (shape, cold, warm) = phases(SUBSET, ASCII, size);
        let glyphs = ASCII.chars().count();

        println!(
            "Latin, {} glyphs at {size:>4} pt:  shape {:>8.1} µs   first draw {:>9.1} µs ({:>6.2} µs/glyph)   per frame {:>8.1} µs ({:>5.2} µs/glyph)",
            glyphs,
            shape.as_secs_f64() * 1e6,
            cold.as_secs_f64() * 1e6,
            micros(cold, glyphs),
            warm.as_secs_f64() * 1e6,
            micros(warm, glyphs),
        );
    }

    // Where the cold cost actually comes from: a fresh renderer pays something once, and each
    // glyph pays again. Three lengths turn that into two numbers instead of one guess.
    println!();

    let alphabet: Vec<char> = ASCII.chars().collect();

    for count in [1_usize, 10, 95] {
        let text: String = alphabet[..count].iter().collect();
        let (_, cold, _) = phases(SUBSET, &text, 22.0);

        println!(
            "Latin, {count:>2} glyph(s) at   22 pt:  first draw {:>9.1} µs",
            cold.as_secs_f64() * 1e6,
        );
    }

    // A CJK font, if one is pointed at: the outlines are an order of magnitude more complicated,
    // which is the thing that makes Chinese expensive to *rasterise* rather than to store.
    if let Ok(path) = std::env::var("POMELO_MEASURE_FONT") {
        let bytes = std::fs::read(&path).expect("the font to measure");
        let size = bytes.len();

        // Leaked: the test process is about to end, and the font system borrows the bytes for
        // the life of the program.
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());

        println!("\n  {} ({} KiB)\n", path, size / 1024,);

        install(bytes);

        // 256 distinct ideographs, which is what a screen of Chinese is worth of them.
        let text: String = (0x4E00u32..0x4E00 + 256)
            .filter_map(char::from_u32)
            .collect();

        for size in [15.0_f32, 22.0] {
            let (shape, first, warm) = phases(bytes, &text, size);
            let glyphs = text.chars().count();

            println!(
                "CJK, {glyphs} glyphs at {size:>4} pt:  shape {:>8.1} µs   first draw {:>9.1} µs ({:>6.2} µs/glyph)   per frame {:>8.1} µs ({:>5.2} µs/glyph)",
                shape.as_secs_f64() * 1e6,
                first.as_secs_f64() * 1e6,
                micros(first, glyphs),
                warm.as_secs_f64() * 1e6,
                micros(warm, glyphs),
            );
        }
    }

    println!();
}
