# iced-pomelo-gfx

iced's renderer contract, implemented over [`pomelo-gfx`](https://github.com/pomelos-on-sale/pomelo-gfx).

The same position in the stack as `iced_tiny_skia` and `iced_wgpu`: it implements the traits iced
asks a renderer for — `iced_core::renderer::Renderer`, and the text, mesh and canvas geometry ones
— and turns a frame of widgets into drawing commands.

What it does *not* own is the pixel buffer. `Renderer` records the frame as a flat list of `Item`s
and hands it over; the platform replays that list into the panel's RGB565 buffer and presents the
damaged regions. That split is iced's own shape rather than an invention here:
`iced_tiny_skia::Renderer::draw` takes the pixel buffer from its caller too, and only its *engine*
rasterises.

```text
iced_widget / iced_runtime      widgets and the runtime
        ↓   the renderer contract — this crate
iced-pomelo-gfx                 records the frame's commands
        ↓   the recorded commands
pomelo-iced-host                replays them into RGB565, presents the damaged regions
        ↓   RGB565
the panel
```

## Where it runs

Inside [pomelo-os](https://github.com/pomelos-on-sale/pomelo-os), on an ESP32-S3 driving a 480×480
AMOLED panel. The platform layer there — `vendor/iced-pomelo-winit`, whose package name is
`iced_winit` because iced's facade insists on that name — depends on this crate and replays what it
records.

`pomelo-os` consumes this repository as a **git dependency** of its patched `iced_renderer`, and
patches that git source back to its own submodule checkout, so a build there uses the same tree as
its `vendor/pomelo-gfx` sibling.

## What is not here yet

Canvas geometry is not behind a feature the way it is in `iced_tiny_skia`, where `geometry` is off
by default and turns on `iced_graphics/geometry`. It is on unconditionally here, because
`geometry.rs` mixes iced's canvas types with the replay helpers that every frame uses
(`color_of`, `solid_paint`, `gradient_paint`). Splitting that module into a `primitive` half and a
canvas half is what would let the feature exist.

## Licence

GPL-3.0-only — see [`LICENSE`](LICENSE).
