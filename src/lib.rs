//! iced's renderer contract, implemented over `pomelo-gfx`.
//!
//! The same position in the stack as `iced_tiny_skia` and `iced_wgpu`: it implements the traits
//! iced asks a renderer for — `iced_core::renderer::Renderer`, and the text, mesh and canvas
//! geometry ones — and turns a frame of widgets into drawing commands.
//!
//! What it does *not* own is the pixel buffer. [`Renderer`] records the frame as a flat list of
//! [`Item`]s and hands it over; `pomelo-iced-host` replays that list into the panel's RGB565
//! buffer and presents the damaged regions. That split is iced's own shape rather than an
//! invention here: `iced_tiny_skia::Renderer::draw` takes the pixel buffer from its caller too,
//! and only its *engine* rasterises.
//!
//! ```text
//! iced_widget / iced_runtime      widgets and the runtime
//!         ↓   the renderer contract — this crate
//! iced-pomelo-gfx                 records the frame's commands
//!         ↓   the recorded commands
//! pomelo-iced-host                replays them into RGB565, presents the damaged regions
//!         ↓   RGB565
//! the panel
//! ```
//!
//! # The recording is one flat list, on purpose
//!
//! iced's own renderer keeps a tree of layers and diffs it to find damage. Here a frame is a
//! `Vec<Item>`, each one already carrying its clip and its placement, and the damage between two
//! frames is a merge-join over two such lists — `pomelo-iced-host`'s `scene` module does that.
//! Flatness is what makes it a merge-join: there is no tree to walk and no index to pair by, so
//! a canvas that redraws a stroke animation appends the strokes that are already finished as
//! *equal* items and damages only the one that is still growing.
//!
//! # Why `pomelo-gfx` is not a drop-in for `tiny-skia`
//!
//! The difference is worth stating precisely: `pomelo-gfx`'s `Canvas::fill_path` *strokes a
//! flattened outline* instead of filling, its `Paint::anti_alias` is stored but never read by the
//! rasteriser, and its clipping stack is rectangular (`Option<Rect>`) where `iced_tiny_skia` uses
//! a coverage `Mask`. What it does have is what these widgets actually need: `fill_rect`,
//! `draw_rrect` / `draw_rrect_stroke`, linear gradients, `blit_mask` for cosmic-text glyph
//! coverage, and the 565 image blits.
//!
//! # What is not here yet
//!
//! Canvas geometry is not behind a feature the way it is in `iced_tiny_skia` (`geometry`,
//! off by default, enabling `iced_graphics/geometry`). It is on unconditionally, because the
//! module mixes iced's canvas types with the replay helpers every frame uses. Splitting
//! `geometry.rs` into `primitive.rs` + `geometry.rs` is the change that would let the feature
//! exist, and it is worth doing when this crate is wired into `iced_renderer` as a backend —
//! there, `iced_widget/canvas` would be what turns it on.

mod renderer;
mod text;

pub mod geometry;

pub use renderer::{Item, Placement, Renderer};
