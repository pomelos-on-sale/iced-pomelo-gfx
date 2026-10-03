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
//!
//! # Two-generation cache and subpixel quantization
//!
//! To prevent severe frame drop spikes on embedded chips when the cache limit is reached, a
//! two-generation cache is used instead of a total wipe. Active glyphs are promoted to the
//! current generation, while unreferenced glyphs naturally age out into the previous generation.
//! Furthermore, subpixel bins in the cache key are quantized to `SubpixelBin::Zero`: on a 1x LCD
//! panel subpixel differences are imperceptible, matching the pre-baked tables and allowing
//! dynamic glyph masks to be reused across all fractional pen positions (cutting RAM and Swash
//! CPU cost by up to 4x).

use std::collections::HashMap;
use std::ops::Deref;

use iced_core::{Color, Point};
use iced_graphics::text::cosmic_text;
use pomelo_gfx::Canvas;

use crate::baked::{Baked, Glyph};
use crate::geometry;

/// Maximum number of dynamic glyph masks retained per generation.
///
/// Across both active and previous generations, this bounds total dynamic glyph RAM to at most
/// `2 * GENERATION_LIMIT` (e.g. 1024 glyphs), matching the previous global cap while providing
/// smooth rotational eviction instead of catastrophic all-at-once drops.
const GENERATION_LIMIT: usize = 512;

/// The rasterised glyphs, by the key of the glyph they came from.
#[derive(Debug)]
pub struct Glyphs {
    /// Active generation of dynamic glyph masks in RAM.
    active_masks: HashMap<cosmic_text::CacheKey, CachedMask>,
    /// Previous generation of dynamic glyph masks, retained to make eviction smooth.
    recent_masks: HashMap<cosmic_text::CacheKey, CachedMask>,
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

/// One rasterised glyph stored in dynamic RAM: its coverage, and where it sits relative to the pen.
#[derive(Debug)]
struct CachedMask {
    coverage: Vec<u8>,
    width: u32,
    height: u32,
    left: i32,
    top: i32,
}

impl CachedMask {
    /// Views this cached mask as a zero-copy [`Glyph`] reference.
    #[inline]
    fn as_glyph(&self) -> Glyph<'_> {
        Glyph {
            coverage: &self.coverage,
            width: self.width,
            height: self.height,
            left: self.left,
            top: self.top,
        }
    }
}

/// One glyph about to be blitted: wrapping its glyph mask and whether it had to be rasterised.
#[derive(Debug, Clone, Copy)]
struct Placed<'a> {
    glyph: Glyph<'a>,
    /// Whether this call had to rasterise it, because it was the first time it had been asked for.
    rasterised: bool,
}

impl<'a> Deref for Placed<'a> {
    type Target = Glyph<'a>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.glyph
    }
}

impl Glyphs {
    /// A cache with nothing in it, and whatever the host baked.
    pub fn new(baked: Option<&'static Baked>) -> Self {
        Self {
            active_masks: HashMap::new(),
            recent_masks: HashMap::new(),
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
    /// Five stages in priority order:
    ///
    /// 1. **The baked tables**, which is a binary search in flash. A glyph found here costs about a
    ///    microsecond instead of the 1.6 ms of rasterising it.
    /// 2. **Active RAM generation**, the hottest working set of dynamic glyphs.
    /// 3. **Recent RAM generation**, promoted to active on hit to keep in-use glyphs alive across rotations.
    /// 4. **Missing glyph (.notdef)**, synthesised as an outline box ("tofu") for visual debugging.
    /// 5. **Swash**, on-demand rasterisation for unbaked glyphs, inserted into the active generation.
    fn mask(
        &mut self,
        font_system: &mut cosmic_text::FontSystem,
        mut key: cosmic_text::CacheKey,
    ) -> Option<Placed<'_>> {
        // Subpixel Bin Quantization:
        // On a 1x display panel, subpixel fractional offsets are visually imperceptible.
        // Quantizing subpixel bins to Zero aligns with the pre-baked tables and allows dynamic
        // Swash masks to be shared across any subpixel pen positions, saving up to 75% RAM
        // and avoiding redundant rasterisation.
        key.x_bin = cosmic_text::SubpixelBin::Zero;
        key.y_bin = cosmic_text::SubpixelBin::Zero;

        // 1. Flash pre-baked tables
        if let Some(glyph) = self.baked.and_then(|baked| baked.glyph(&key)) {
            return Some(Placed {
                glyph,
                rasterised: false,
            });
        }

        // 2. Active generation cache hit
        if self.active_masks.contains_key(&key) {
            let mask = self.active_masks.get(&key)?;
            return Some(Placed {
                glyph: mask.as_glyph(),
                rasterised: false,
            });
        }

        // 3. Recent generation cache hit (promote to active generation)
        if let Some(mask) = self.recent_masks.remove(&key) {
            if self.active_masks.len() >= GENERATION_LIMIT {
                self.recent_masks = std::mem::take(&mut self.active_masks);
            }
            self.active_masks.insert(key, mask);
            let mask = self.active_masks.get(&key)?;
            return Some(Placed {
                glyph: mask.as_glyph(),
                rasterised: false,
            });
        }

        // 4. Missing glyph (.notdef, glyph_id == 0): synthesize a tofu (box) mask for visual debugging
        if key.glyph_id == 0 {
            if self.active_masks.len() >= GENERATION_LIMIT {
                self.recent_masks = std::mem::take(&mut self.active_masks);
            }
            let tofu = synthesize_tofu_mask(key.font_size_bits);
            self.active_masks.insert(key, tofu);
            let mask = self.active_masks.get(&key)?;
            return Some(Placed {
                glyph: mask.as_glyph(),
                rasterised: true,
            });
        }

        // 5. Swash fallback: rasterise on the device
        let raster = crate::profile::start(crate::profile::Phase::Rasterise);

        let image = match self.swash.get_image_uncached(font_system, key) {
            Some(img) => img,
            None => {
                drop(raster);
                if self.active_masks.len() >= GENERATION_LIMIT {
                    self.recent_masks = std::mem::take(&mut self.active_masks);
                }
                let tofu = synthesize_tofu_mask(key.font_size_bits);
                self.active_masks.insert(key, tofu);
                let mask = self.active_masks.get(&key)?;
                return Some(Placed {
                    glyph: mask.as_glyph(),
                    rasterised: true,
                });
            }
        };

        drop(raster);

        if self.active_masks.len() >= GENERATION_LIMIT {
            self.recent_masks = std::mem::take(&mut self.active_masks);
        }

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

        self.active_masks.insert(
            key,
            CachedMask {
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

        let mask = self.active_masks.get(&key)?;

        Some(Placed {
            glyph: mask.as_glyph(),
            rasterised: true,
        })
    }
}

/// Synthesizes a rectangular outline mask ("tofu" / replacement box) for missing glyphs (`.notdef`).
fn synthesize_tofu_mask(font_size_bits: u32) -> CachedMask {
    let size = f32::from_bits(font_size_bits);
    let size = if size.is_finite() && size > 0.0 {
        size
    } else {
        14.0
    };

    let width = (size * 0.55).round().max(4.0) as u32;
    let height = (size * 0.75).round().max(5.0) as u32;
    let top = (size * 0.75).round() as i32;
    let left = 1i32;

    let mut coverage = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            if y == 0 || y == height - 1 || x == 0 || x == width - 1 {
                coverage[(y * width + x) as usize] = 255;
            }
        }
    }

    CachedMask {
        coverage,
        width,
        height,
        left,
        top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placed_deref_to_glyph() {
        let coverage = [255u8; 16];
        let glyph = Glyph {
            coverage: &coverage,
            width: 4,
            height: 4,
            left: 1,
            top: 2,
        };
        let placed = Placed {
            glyph,
            rasterised: false,
        };

        assert_eq!(placed.width, 4);
        assert_eq!(placed.height, 4);
        assert_eq!(placed.left, 1);
        assert_eq!(placed.top, 2);
        assert_eq!(placed.coverage.len(), 16);
        assert!(!placed.rasterised);
    }

    #[test]
    fn cached_mask_as_glyph() {
        let mask = CachedMask {
            coverage: vec![128; 8],
            width: 2,
            height: 4,
            left: -1,
            top: 3,
        };
        let glyph = mask.as_glyph();
        assert_eq!(glyph.width, 2);
        assert_eq!(glyph.height, 4);
        assert_eq!(glyph.left, -1);
        assert_eq!(glyph.top, 3);
        assert_eq!(glyph.coverage, &[128; 8]);
    }

    #[test]
    fn two_generation_eviction_and_promotion() {
        let mut glyphs = Glyphs::new(None);

        let font_system = iced_graphics::text::font_system();
        let mut font_system = font_system.write().expect("font system");

        // Request a glyph with different subpixel bins — both should hit the same quantized key
        let mut buffer = cosmic_text::Buffer::new(font_system.raw(), cosmic_text::Metrics::new(16.0, 20.0));
        buffer.set_text(
            font_system.raw(),
            "A",
            &cosmic_text::Attrs::new(),
            cosmic_text::Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(font_system.raw(), false);

        let runs = buffer.layout_runs().collect::<Vec<_>>();
        assert!(!runs.is_empty());
        let glyph = &runs[0].glyphs[0];

        // 1st request at (0.0, 0.0) -> rasterised
        let p1 = glyph.physical((0.0, 0.0), 1.0);
        let (w1, h1) = {
            let m1 = glyphs.mask(font_system.raw(), p1.cache_key).expect("mask exists");
            assert!(m1.rasterised, "first time should be rasterised");
            (m1.width, m1.height)
        };

        // 2nd request at (0.4, 0.7) -> different subpixel bin, but should hit the same quantized cache!
        let p2 = glyph.physical((0.4, 0.7), 1.0);
        assert_ne!(p1.cache_key.x_bin, p2.cache_key.x_bin);
        {
            let m2 = glyphs.mask(font_system.raw(), p2.cache_key).expect("mask exists");
            assert!(!m2.rasterised, "subpixel quantization must hit RAM cache without re-rasterising");
            assert_eq!(w1, m2.width);
            assert_eq!(h1, m2.height);
        }

        // Verify active masks contains 1 entry
        assert_eq!(glyphs.active_masks.len(), 1);
        assert_eq!(glyphs.recent_masks.len(), 0);

        // Fill active generation up to limit with synthetic keys
        let base_key = p1.cache_key;
        for i in 1..GENERATION_LIMIT {
            let mut k = base_key;
            k.glyph_id = i as u16 + 1000;
            glyphs.active_masks.insert(
                k,
                CachedMask {
                    coverage: Vec::new(),
                    width: 0,
                    height: 0,
                    left: 0,
                    top: 0,
                },
            );
        }
        assert_eq!(glyphs.active_masks.len(), GENERATION_LIMIT);

        // Shape 'B' at 24px as a distinct glyph
        let mut b_buffer = cosmic_text::Buffer::new(font_system.raw(), cosmic_text::Metrics::new(24.0, 28.0));
        b_buffer.set_text(
            font_system.raw(),
            "B",
            &cosmic_text::Attrs::new(),
            cosmic_text::Shaping::Advanced,
            None,
        );
        b_buffer.shape_until_scroll(font_system.raw(), false);
        let b_glyph = &b_buffer.layout_runs().next().unwrap().glyphs[0];
        let p_b = b_glyph.physical((0.0, 0.0), 1.0);

        // Inserting 'B' into full active generation triggers rotation
        {
            let m_b = glyphs.mask(font_system.raw(), p_b.cache_key).expect("mask exists");
            assert!(m_b.rasterised);
        }

        // Now: active_masks has 'B', recent_masks has the previous generation (including 'A')
        assert_eq!(glyphs.active_masks.len(), 1);
        assert_eq!(glyphs.recent_masks.len(), GENERATION_LIMIT);

        // Accessing 'A' again should promote it from recent_masks into active_masks without re-rasterising!
        {
            let m3 = glyphs.mask(font_system.raw(), p1.cache_key).expect("mask exists");
            assert!(!m3.rasterised, "promoted glyph from recent generation must not be re-rasterised");
        }
        assert_eq!(glyphs.active_masks.len(), 2);
        assert_eq!(glyphs.recent_masks.len(), GENERATION_LIMIT - 1);
    }

    #[test]
    fn synthesize_tofu_mask_creates_valid_outline_box() {
        let mask = synthesize_tofu_mask(18.0f32.to_bits());
        assert_eq!(mask.width, 10);
        assert_eq!(mask.height, 14);
        assert_eq!(mask.top, 14);
        assert_eq!(mask.left, 1);
        assert_eq!(mask.coverage.len(), (10 * 14) as usize);

        // Check top and bottom border are fully covered
        for x in 0..10 {
            assert_eq!(mask.coverage[x], 255, "top border at x={x}");
            assert_eq!(mask.coverage[13 * 10 + x], 255, "bottom border at x={x}");
        }

        // Check left and right border are fully covered
        for y in 0..14 {
            assert_eq!(mask.coverage[y * 10], 255, "left border at y={y}");
            assert_eq!(mask.coverage[y * 10 + 9], 255, "right border at y={y}");
        }

        // Check interior is hollow (0 coverage)
        for y in 1..13 {
            for x in 1..9 {
                assert_eq!(mask.coverage[y * 10 + x], 0, "interior at ({x}, {y}) must be empty");
            }
        }
    }

    #[test]
    fn missing_glyph_zero_returns_tofu_mask_and_caches() {
        let mut glyphs = Glyphs::new(None);
        let font_system = iced_graphics::text::font_system();
        let mut font_system = font_system.write().expect("font system");

        let mut buffer = cosmic_text::Buffer::new(font_system.raw(), cosmic_text::Metrics::new(18.0, 22.0));
        buffer.set_text(
            font_system.raw(),
            "X",
            &cosmic_text::Attrs::new(),
            cosmic_text::Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(font_system.raw(), false);
        let base_glyph = &buffer.layout_runs().next().unwrap().glyphs[0];
        let mut key = base_glyph.physical((0.0, 0.0), 1.0).cache_key;
        key.glyph_id = 0; // .notdef

        // 1st request -> synthesised tofu mask, rasterised=true
        {
            let m1 = glyphs.mask(font_system.raw(), key).expect("tofu mask must be returned");
            assert!(m1.rasterised);
            assert_eq!(m1.width, 10);
            assert_eq!(m1.height, 14);
        }

        // 2nd request -> RAM cache hit, rasterised=false
        {
            let m2 = glyphs.mask(font_system.raw(), key).expect("cached tofu mask");
            assert!(!m2.rasterised);
            assert_eq!(m2.width, 10);
            assert_eq!(m2.height, 14);
        }
    }
}

