//! The pre-baked glyph tables, and the one place a renderer asks them for a mask.
//!
//! Rasterising a glyph on this board costs about **1.6 ms** — a screen with 147 of them (the
//! settings app opening) is 238 ms of a 400 ms frame. The glyphs are the same every time and the
//! sizes are known at build time, so they are rasterised on the build machine instead
//! (`tools/bake_glyphs.py` in `pomelo-os`) and the device only looks them up. That is the whole
//! idea: 1.6 ms of ESP32-S3 arithmetic becomes a binary search over a table in flash.
//!
//! What is *not* baked is anything that is not pixels: advances, kerning and line breaking stay
//! cosmic-text's business, so the font itself still ships and shaping still happens on the device.
//! A size with no table, and a glyph no table has, fall through to swash exactly as before — the
//! tables are a cache that happens to be pre-filled, not a requirement.
//!
//! # The table format (`tools/bake_glyphs.py` writes it)
//!
//! ```text
//! header, 40 bytes, little-endian:
//!   0   magic           4    b"PGLY"
//!   4   version         u16  1
//!   6   flags           u16  0
//!   8   font_size_bits  u32  the size's f32 bits — compared bit for bit, never as a float
//!   12  font_hash       16   SHA-256 of the font it was baked from, first 16 bytes
//!   28  glyph_count     u32
//!   32  coverage_len    u32
//!   36  reserved        u32
//!
//! glyph records, 12 bytes each, glyph_count of them, sorted by glyph_id:
//!   0   glyph_id    u16
//!   2   width       u8
//!   3   height      u8
//!   4   left        i8
//!   5   top         i8
//!   6   reserved    u8
//!   7   reserved    u8
//!   8   offset      u32   into the coverage pool
//!
//! coverage pool: width * height bytes per glyph, row-major, 8-bit alpha.
//! ```
//!
//! # Why the lookup ignores the subpixel bins
//!
//! `CacheKey` carries `x_bin`/`y_bin` — where the pen sits inside a pixel — and swash rasterises a
//! separate mask for each bin. The tables are baked at bin zero: one mask per glyph and size, which
//! is what makes them small enough to ship. The blit position is the same either way (`physical()`
//! hands out integers), so the only difference is that a baked glyph is drawn on the pixel grid
//! while a swash one is drawn a fraction of a pixel off. On a 1x panel that is invisible.
//!
//! What *is* checked, because it would draw the wrong picture: the face (`font_id` — a bold or
//! italic face is a different one, and iced switches faces rather than synthesising weight), the
//! size, and the absence of a synthetic slant.

use std::cmp::Ordering;
use std::fmt;
use std::sync::OnceLock;

use iced_graphics::text::cosmic_text::{self, CacheKey, CacheKeyFlags};

/// Bytes of header before the first glyph record.
const HEADER: usize = 40;

/// Bytes of one glyph record.
const RECORD: usize = 12;

const MAGIC: &[u8; 4] = b"PGLY";
const VERSION: u16 = 1;

/// Why a table could not be used.
///
/// These are build-time mistakes — a table is `include_bytes!` data, so a bad one means the image
/// was built wrong — and the answer to them is to say which one it is rather than to carry on
/// drawing something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The bytes are not a table at all (wrong magic, or something else entirely).
    NotATable,
    /// A version this runtime does not know.
    UnsupportedVersion(u16),
    /// The header is there but the records and coverage are not.
    Truncated { declared: usize, got: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotATable => formatter.write_str("not a baked glyph table (no PGLY magic)"),
            Self::UnsupportedVersion(version) => write!(
                formatter,
                "baked glyph table version {version}, which this runtime does not know"
            ),
            Self::Truncated { declared, got } => write!(
                formatter,
                "baked glyph table declares {declared} B of records and coverage, but has {got} B"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// One baked table: one font size's worth of glyphs.
///
/// It borrows the bytes it was parsed from, so it is only ever as long-lived as the `include_bytes!`
/// it came from — which is the point: nothing is copied out of flash.
#[derive(Debug, Clone, Copy)]
pub struct Atlas {
    bytes: &'static [u8],
    count: u32,
}

impl Atlas {
    /// Parses a table, checking it against itself.
    ///
    /// The checks are cheap and they are the ones that catch a real mistake: the magic, the version,
    /// and that the bytes actually hold as many records and coverage as the header claims.
    pub fn new(bytes: &'static [u8]) -> Result<Self, Error> {
        if bytes.len() < HEADER || &bytes[0..4] != MAGIC {
            return Err(Error::NotATable);
        }

        let version = u16::from_le_bytes([bytes[4], bytes[5]]);

        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }

        let count = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let coverage = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]);
        let declared = HEADER + count as usize * RECORD + coverage as usize;

        if bytes.len() != declared {
            return Err(Error::Truncated {
                declared,
                got: bytes.len(),
            });
        }

        Ok(Self { bytes, count })
    }

    /// The size this table was baked at, as the `f32` bits the cache keys with.
    pub fn font_size_bits(&self) -> u32 {
        u32::from_le_bytes([self.bytes[8], self.bytes[9], self.bytes[10], self.bytes[11]])
    }

    /// The first 16 bytes of the source font's SHA-256, for provenance.
    pub fn font_hash(&self) -> &'static [u8] {
        &self.bytes[12..28]
    }

    /// How many glyphs the table carries.
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// Whether the table carries no glyphs at all.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The mask for one glyph, or `None` if this table does not have it.
    ///
    /// A binary search, because the records are sorted by `glyph_id` — 4008 of them is twelve
    /// comparisons, and that is the entire per-glyph cost of the baked path.
    pub fn glyph(&self, glyph_id: u16) -> Option<Glyph<'_>> {
        let (mut low, mut high) = (0, self.count);

        while low < high {
            let middle = low + (high - low) / 2;
            let record = self.record(middle);
            let id = u16::from_le_bytes([record[0], record[1]]);

            match id.cmp(&glyph_id) {
                Ordering::Less => low = middle + 1,
                Ordering::Greater => high = middle,
                Ordering::Equal => return Some(self.glyph_at(record)),
            }
        }

        None
    }

    /// The record at `index`, which the caller has already bounds-checked by construction.
    fn record(&self, index: u32) -> &'static [u8] {
        let start = HEADER + index as usize * RECORD;

        &self.bytes[start..start + RECORD]
    }

    /// Reads a record into the shape the blitter wants.
    fn glyph_at(&self, record: &'static [u8]) -> Glyph<'static> {
        let width = u32::from(record[2]);
        let height = u32::from(record[3]);
        let offset = u32::from_le_bytes([record[8], record[9], record[10], record[11]]);
        let coverage = HEADER + self.count as usize * RECORD;
        let start = coverage + offset as usize;

        Glyph {
            coverage: &self.bytes[start..start + (width * height) as usize],
            width,
            height,
            left: i32::from(record[4] as i8),
            top: i32::from(record[5] as i8),
        }
    }
}

/// One glyph's coverage and where it sits relative to the pen.
///
/// Borrowed from the table, so nothing is allocated and nothing is copied out of flash: the same
/// shape [`Mask`](crate::Renderer) uses for the glyphs that had to be rasterised at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glyph<'a> {
    /// 8-bit alpha, row-major, `width * height` bytes.
    pub coverage: &'a [u8],
    pub width: u32,
    pub height: u32,
    /// Horizontal offset from the pen, in pixels.
    pub left: i32,
    /// Vertical offset from the baseline, in pixels, positive upwards.
    pub top: i32,
}

/// Every baked table for one font, by size.
#[derive(Debug, Clone)]
pub struct Baked {
    font_id: cosmic_text::fontdb::ID,
    tables: Vec<Atlas>,
}

impl Baked {
    /// Parses a set of tables and binds them to the face they were baked from.
    pub fn new(
        font_id: cosmic_text::fontdb::ID,
        tables: &'static [&'static [u8]],
    ) -> Result<Self, Error> {
        let tables = tables
            .iter()
            .map(|bytes| Atlas::new(bytes))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { font_id, tables })
    }

    /// The face these tables are for.
    pub fn font_id(&self) -> cosmic_text::fontdb::ID {
        self.font_id
    }

    /// The tables, one per size.
    pub fn tables(&self) -> &[Atlas] {
        &self.tables
    }

    /// The baked mask for one cache key, if there is one.
    ///
    /// `None` is the ordinary answer and not a failure: it means this glyph, or this size, is not
    /// in any table, and the caller rasterises it as it always did.
    pub fn glyph(&self, key: &CacheKey) -> Option<Glyph<'_>> {
        // A synthetic slant is drawn by the rasteriser, not by a table, so a table has nothing to
        // say about it. Bold is not here on purpose: that is a different face, and a different face
        // is a different `font_id`, which the check below already refuses.
        if key.flags.contains(CacheKeyFlags::FAKE_ITALIC) || key.font_id != self.font_id {
            return None;
        }

        self.tables
            .iter()
            .find(|table| table.font_size_bits() == key.font_size_bits)
            .and_then(|table| table.glyph(key.glyph_id))
    }
}

/// The tables the host installed, if it installed any.
static INSTALLED: OnceLock<Baked> = OnceLock::new();

/// Installs the pre-baked tables for `font_id`.
///
/// Called by the platform layer at the moment it installs the font the tables were baked from —
/// which is why the face is named here and not guessed: a table's glyph numbers only mean anything
/// against the one font it was baked from.
///
/// # Panics
///
/// If a table is not parseable. These are `include_bytes!` assets, so a broken one is a build that
/// should never have shipped, and the honest answer is to say which table is broken rather than to
/// draw with the glyphs that happen to parse. The tables are not a requirement: installing none
/// costs the milliseconds this module exists to save, and nothing else.
pub fn install(font_id: cosmic_text::fontdb::ID, tables: &'static [&'static [u8]]) {
    let baked = Baked::new(font_id, tables)
        .unwrap_or_else(|error| panic!("the baked glyph tables cannot be used: {error}"));

    // A second install is a no-op: the first one is the font the host chose, and the platform
    // installs its default once.
    let _ = INSTALLED.set(baked);
}

/// The installed tables, for a renderer asking for a mask.
pub fn get() -> Option<&'static Baked> {
    INSTALLED.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One glyph as the baker writes it: id, width, height, left, top, coverage.
    type BakedGlyph = (u16, u8, u8, i8, i8, &'static [u8]);

    /// Builds a table the way the baker does, so the parser is tested against the real format.
    fn table(size: f32, glyphs: &[BakedGlyph]) -> &'static [u8] {
        let mut sorted = glyphs.to_vec();
        sorted.sort_by_key(|glyph| glyph.0);

        let mut coverage = Vec::new();
        let mut records = Vec::new();

        for (id, width, height, left, top, bytes) in sorted {
            let offset = coverage.len() as u32;
            coverage.extend_from_slice(bytes);

            records.extend_from_slice(&id.to_le_bytes());
            records.push(width);
            records.push(height);
            records.push(left as u8);
            records.push(top as u8);
            records.push(0);
            records.push(0);
            records.extend_from_slice(&offset.to_le_bytes());
        }

        let mut table = Vec::new();
        table.extend_from_slice(MAGIC);
        table.extend_from_slice(&VERSION.to_le_bytes());
        table.extend_from_slice(&0u16.to_le_bytes());
        table.extend_from_slice(&size.to_bits().to_le_bytes());
        table.extend_from_slice(&[7u8; 16]);
        table.extend_from_slice(&(glyphs.len() as u32).to_le_bytes());
        table.extend_from_slice(&(coverage.len() as u32).to_le_bytes());
        table.extend_from_slice(&0u32.to_le_bytes());
        table.extend_from_slice(&records);
        table.extend_from_slice(&coverage);

        Box::leak(table.into_boxed_slice())
    }

    #[test]
    fn a_table_parses_and_answers_by_glyph_id() {
        let bytes = table(
            16.0,
            &[(9, 2, 2, 1, 3, &[10, 20, 30, 40]), (3, 1, 1, 0, 0, &[255])],
        );

        let atlas = Atlas::new(bytes).expect("a table this crate just built");

        assert_eq!(atlas.font_size_bits(), 16.0f32.to_bits());
        assert_eq!(atlas.len(), 2);
        assert_eq!(atlas.font_hash(), &[7u8; 16]);

        let glyph = atlas.glyph(9).expect("a glyph the table has");

        assert_eq!((glyph.width, glyph.height), (2, 2));
        assert_eq!((glyph.left, glyph.top), (1, 3));
        assert_eq!(glyph.coverage, &[10, 20, 30, 40]);

        let single = atlas.glyph(3).expect("the other glyph");

        assert_eq!(single.coverage, &[255]);
        assert_eq!((single.left, single.top), (0, 0));

        assert!(atlas.glyph(4).is_none(), "a glyph it does not have");
        assert!(atlas.glyph(0).is_none(), "below the first id");
        assert!(atlas.glyph(u16::MAX).is_none(), "above the last id");
    }

    #[test]
    fn negative_bearings_survive_the_signed_bytes() {
        let bytes = table(18.0, &[(1, 1, 1, -3, -5, &[1])]);
        let atlas = Atlas::new(bytes).expect("a table");

        let glyph = atlas.glyph(1).expect("the glyph");

        assert_eq!((glyph.left, glyph.top), (-3, -5));
    }

    #[test]
    fn a_table_that_is_not_one_is_refused() {
        assert_eq!(Atlas::new(b"nope").err(), Some(Error::NotATable));
        assert_eq!(Atlas::new(&[]).err(), Some(Error::NotATable));
    }

    #[test]
    fn an_unknown_version_is_refused() {
        let mut bytes = table(16.0, &[(1, 1, 1, 0, 0, &[1])]).to_vec();
        bytes[4..6].copy_from_slice(&2u16.to_le_bytes());

        assert_eq!(
            Atlas::new(Box::leak(bytes.into_boxed_slice())).err(),
            Some(Error::UnsupportedVersion(2))
        );
    }

    #[test]
    fn a_table_that_lost_its_coverage_is_refused() {
        let bytes = table(16.0, &[(1, 2, 2, 0, 0, &[1, 2, 3, 4])]);
        let short = &bytes[..bytes.len() - 2];

        assert_eq!(
            Atlas::new(Box::leak(short.to_vec().into_boxed_slice())).err(),
            Some(Error::Truncated {
                declared: bytes.len(),
                got: bytes.len() - 2,
            })
        );
    }
}
