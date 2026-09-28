//! Drawing shaped text into RGB565: the glyph masks, and the blit.
//!
//! iced hands a renderer a *shaped* paragraph — the widget that owns the text keeps a
//! `iced_graphics::text::Paragraph` in its own tree state and re-shapes it only when the text
//! changes — so what is left here is rasterising glyphs and blending them. `cosmic_text` does the
//! rasterising (`SwashCache`), and `pomelo-gfx` does the blending (`Canvas::blit_mask`, which is a
//! mask-and-a-colour blit and exists for exactly this).
//!
//! What is ours is the cache, and it earns its place: a glyph is rasterised once per (glyph, size)
//! and re-blitted in whatever colour and at whatever position each frame, which is what keeps a
//! frame that only moved a label from rasterising its letters again.
//!
//! # Why the masks are cached without the colour
//!
//! `iced_tiny_skia` bakes the colour into the cached glyph and keys on `(glyph, r, g, b)`, because
//! its blit takes premultiplied RGBA. Ours takes a coverage mask and a colour, so the mask alone is
//! the cacheable part and one series of glyphs serves every colour — the same white label and the
//! same grey label share every letter.

use std::collections::HashMap;

use iced_core::{Color, Point};
use iced_graphics::text::cosmic_text;
use pomelo_gfx::Canvas;

use crate::geometry;

/// The rasterised glyphs, by the key of the glyph they came from.
#[derive(Debug)]
pub struct Glyphs {
    masks: HashMap<cosmic_text::CacheKey, Mask>,
    swash: cosmic_text::SwashCache,
}

impl Default for Glyphs {
    fn default() -> Self {
        Self {
            masks: HashMap::new(),
            // `SwashCache` is not `Default`: it allocates the rasteriser's own scratch.
            swash: cosmic_text::SwashCache::new(),
        }
    }
}

/// One rasterised glyph: its coverage, and where it sits relative to the pen.
#[derive(Debug)]
struct Mask {
    coverage: Vec<u8>,
    width: u32,
    height: u32,
    left: i32,
    top: i32,
}

impl Glyphs {
    /// A cache with nothing in it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Draws a shaped buffer with its top-left at `position`, in `color`.
    ///
    /// The canvas's clip is what limits this: each glyph is blitted through it, so a glyph half
    /// outside the damage costs half a glyph.
    pub fn draw(
        &mut self,
        canvas: &mut Canvas<'_>,
        font_system: &mut cosmic_text::FontSystem,
        buffer: &cosmic_text::Buffer,
        position: Point,
        color: Color,
    ) {
        for run in buffer.layout_runs() {
            for glyph in run.glyphs {
                let physical = glyph.physical((position.x, position.y), 1.0);

                // A span can carry its own colour; without one the run's colour is the caller's.
                // iced states a colour as three bytes of RGB and an alpha in `0.0..=1.0`.
                let color = glyph.color_opt.map_or(color, |color| {
                    let [r, g, b, a] = color.as_rgba();

                    Color::from_rgba8(r, g, b, a as f32 / 255.0)
                });

                let Some(mask) = self.mask(font_system, physical.cache_key) else {
                    continue;
                };

                canvas.blit_mask(
                    physical.x + mask.left,
                    physical.y - mask.top + run.line_y.round() as i32,
                    mask.width,
                    mask.height,
                    &mask.coverage,
                    geometry::color_of(color),
                );
            }
        }
    }

    /// The mask for one glyph, rasterising it if this is the first time it has been asked for.
    fn mask(
        &mut self,
        font_system: &mut cosmic_text::FontSystem,
        key: cosmic_text::CacheKey,
    ) -> Option<&Mask> {
        if !self.masks.contains_key(&key) {
            // A cap, and a crude one: when it is reached the whole cache goes, which costs one
            // frame of re-rasterising and keeps the growth bounded. The UI has one size per widget,
            // so the working set is a few hundred glyphs and this is not reached in practice —
            // `iced_tiny_skia` trims least-recently-used instead, which is the upgrade if it
            // ever matters.
            const LIMIT: usize = 1024;

            if self.masks.len() >= LIMIT {
                self.masks.clear();
            }

            let image = self.swash.get_image_uncached(font_system, key)?;

            // Only a coverage mask can be blitted by `blit_mask`. A coloured glyph (an emoji) and
            // a subpixel one are both richer than that and would need an RGBA blit the rasteriser
            // does not have; neither can come from the 16 KiB Latin subset this OS embeds.
            if !matches!(image.content, cosmic_text::SwashContent::Mask)
                || image.placement.width == 0
                || image.placement.height == 0
            {
                return None;
            }

            self.masks.insert(
                key,
                Mask {
                    coverage: image.data.clone(),
                    width: image.placement.width,
                    height: image.placement.height,
                    left: image.placement.left,
                    top: image.placement.top,
                },
            );
        }

        self.masks.get(&key)
    }
}
