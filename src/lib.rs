//! iced's renderer contract, implemented over `pomelo-gfx`.
//!
//! The same position in the stack as `iced_tiny_skia` and `iced_wgpu`: it implements the traits
//! iced asks a renderer for — `iced_core::renderer::Renderer`, and the text, mesh and canvas
//! geometry ones — and turns a frame of widgets into drawing commands.
//!
//! What it does *not* own is the platform: iced's `Compositor` contract is implemented here (a
//! [`Surface`] owns the panel's RGB565 frame buffer and replays the recording into it), but where
//! those pixels go from there — a QSPI panel, a fake in a test — is the host's business.
//!
//! ```text
//! iced_widget / iced_runtime      widgets and the runtime
//!         ↓   the renderer contract — this crate
//! iced-pomelo-gfx                 records the frame's commands, replays them into RGB565
//!         ↓   RGB565, plus the damaged rectangles
//! pomelo-iced-host                hands them to the panel, and polls the touch
//!         ↓   RGB565
//! the panel
//! ```
//!
//! # The recording is iced's, and the damage inside it is finer
//!
//! A frame is a stack of [`Layer`]s — iced's shape, held in iced's own
//! [`Stack`](iced_graphics::layer::Stack): quads, canvas geometry and text, each diffed where
//! `iced_tiny_skia` diffs it, with the `Live`/`Group`/`Cached` cases a `canvas::Cache` needs.
//! What differs is one function: [`Layer::damage`] pairs a layer's commands by geometry instead
//! of by index, and walks a canvas's own recording command by command. Both are refinements of
//! iced's answer — never coarser — and the [`layer`] module says where and why.
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

pub mod baked;
pub mod compositor;
pub mod geometry;
pub mod layer;
pub mod panel;
pub mod profile;
pub mod surface;

pub use compositor::{Compositor, Panel};
pub use layer::{Item, Layer, Text};
pub use renderer::{Placement, Renderer};
pub use surface::Surface;
