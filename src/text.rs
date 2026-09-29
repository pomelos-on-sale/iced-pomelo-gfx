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

use crate::baked::Baked;
use crate::geometry;

/// The rasterised glyphs, by the key of the glyph they came from.
#[derive(Debug)]
pub struct Glyphs {
    /// The glyphs that had to be rasterised on the device — the fallback, and only ever the ones a
    /// baked table did not have. A screen whose sizes are all baked leaves this empty.
    masks: HashMap<cosmic_text::CacheKey, Mask>,
    swash: cosmic_text::SwashCache,
    /// The pre-baked tables the host installed, if it did. Consulted first, and the reason a screen
    /// full of text costs what it costs.
    baked: Option<&'static Baked>,
}

impl Default for Glyphs {
    fn default() -> Self {
        Self::new(None)
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

/// One glyph about to be blitted: where its coverage came from is the caller's business.
#[derive(Debug)]
struct Placed<'a> {
    coverage: &'a [u8],
    width: u32,
    height: u32,
    left: i32,
    top: i32,
    /// Whether this call had to rasterise it, because it was the first time it had been asked for.
    rasterised: bool,
}

impl Glyphs {
    /// A cache with nothing in it, and whatever the host baked.
    pub fn new(baked: Option<&'static Baked>) -> Self {
        Self {
            masks: HashMap::new(),
            // `SwashCache` is not `Default`: it allocates the rasteriser's own scratch.
            swash: cosmic_text::SwashCache::new(),
            baked,
        }
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

                let _blit = crate::profile::start(crate::profile::Phase::Blits);

                canvas.blit_mask(
                    physical.x + mask.left,
                    physical.y - mask.top + run.line_y.round() as i32,
                    mask.width,
                    mask.height,
                    mask.coverage,
                    geometry::color_of(color),
                );

                // One glyph, counted once: the blit is what happened, and whether the mask had to
                // be rasterised for it is the part of the text cost that a screen pays only once.
                crate::profile::glyph(mask.rasterised);
            }
        }
    }

    /// The mask for one glyph, wherever it comes from.
    ///
    /// Three sources, in this order, and the order is the whole optimisation:
    ///
    /// 1. **The baked tables**, which is a binary search in flash. A glyph found here costs about a
    ///    microsecond instead of the 1.6 ms of rasterising it.
    /// 2. **The mask cache**, which holds only what had to be rasterised, so it stays small — the
    ///    baked sizes never put anything in it at all.
    /// 3. **Swash**, which rasterises it and is the reason nothing is ever missing: a size with no
    ///    table and a glyph no table has both land here, and both are correct.
    fn mask(
        &mut self,
        font_system: &mut cosmic_text::FontSystem,
        key: cosmic_text::CacheKey,
    ) -> Option<Placed<'_>> {
        if let Some(glyph) = self.baked.and_then(|baked| baked.glyph(&key)) {
            return Some(Placed {
                coverage: glyph.coverage,
                width: glyph.width,
                height: glyph.height,
                left: glyph.left,
                top: glyph.top,
                rasterised: false,
            });
        }

        let rasterised = if !self.masks.contains_key(&key) {
            // A cap, and a crude one: when it is reached the whole cache goes, which costs one
            // frame of re-rasterising and keeps the growth bounded. The UI has one size per widget,
            // so the working set is a few hundred glyphs and this is not reached in practice —
            // `iced_tiny_skia` trims least-recently-used instead, which is the upgrade if it
            // ever matters.
            const LIMIT: usize = 1024;

            if self.masks.len() >= LIMIT {
                self.masks.clear();
            }

            // The one phase a screen pays only once: the mask for this glyph does not exist yet,
            // so it is rasterised here, and every later frame that draws the same glyph blits it.
            let raster = crate::profile::start(crate::profile::Phase::Rasterise);

            let image = self.swash.get_image_uncached(font_system, key)?;

            drop(raster);

            // Only a coverage mask can be blitted by `blit_mask`. A coloured glyph (an emoji) and
            // a subpixel one are both richer than that and would need an RGBA blit the rasteriser
            // does not have; neither can come from the subset this OS embeds. A glyph with no
            // placement draws nothing at all, which is a space.
            //
            // All three are cached as a mask with no coverage rather than passed over, and the
            // difference is not cosmetic: a Settings screen has twenty-two of them, and a glyph
            // that is not cached is rasterised again by *every* frame that draws it.
            let nothing = !matches!(image.content, cosmic_text::SwashContent::Mask)
                || image.placement.width == 0
                || image.placement.height == 0;

            self.masks.insert(
                key,
                Mask {
                    coverage: if nothing {
                        Vec::new()
                    } else {
                        image.data.clone()
                    },
                    width: if nothing { 0 } else { image.placement.width },
                    height: if nothing { 0 } else { image.placement.height },
                    left: image.placement.left,
                    top: image.placement.top,
                },
            );

            true
        } else {
            false
        };

        let mask = self.masks.get(&key)?;

        Some(Placed {
            coverage: &mask.coverage,
            width: mask.width,
            height: mask.height,
            left: mask.left,
            top: mask.top,
            rasterised,
        })
    }
}
