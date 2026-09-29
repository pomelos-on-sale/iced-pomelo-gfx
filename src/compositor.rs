//! iced's compositor contract, implemented for a panel that is not a window.
//!
//! A `Program` names its renderer, and `iced_program::Renderer` — the bound that trait states —
//! asks that renderer for a [`compositor::Default`](iced_graphics::compositor::Default). That is
//! how iced gets from "an app's view is drawn by this renderer" to "somewhere to put the pixels":
//! the shell creates `<Renderer as Default>::Compositor`, asks it for a renderer and a surface,
//! and calls `present` once per frame. On a desktop the compositor owns a window's buffer (through
//! softbuffer or wgpu); here it owns the panel's.
//!
//! This is the type that makes our shell's `run(program)` need no renderer bound at all: iced's
//! own `iced_winit::run` is generic over `P: Program` and reaches the renderer through this
//! contract, which is the only way a `Program` handed in by iced's facade — its wrapper is generic
//! over `P::Renderer` — can be accepted.
//!
//! # Where the panel comes from
//!
//! [`panel::set`](crate::panel::set), not the constructor: `Compositor::new` takes a *display
//! connection*, and this board has no window handle to give. See [`Panel`] and the module docs
//! there.
//!
//! # What is fixed and what is not
//!
//! The panel is 1:1 and never resizes, so the viewport a surface was made with is the one every
//! frame is drawn for; the `viewport` argument of `present` is accepted and ignored, because
//! there is nothing a caller could say with it that this hardware would act on.

use iced_core::Color;
use iced_graphics::compositor::{Information, SurfaceError};
use iced_graphics::error::Reason;
use iced_graphics::{Error, Settings, Shell, Viewport};
use raw_window_handle::{HandleError, HasDisplayHandle, HasWindowHandle};

use crate::panel::{self, Sink};
use crate::{Renderer, Surface};

/// The display connection iced's compositor is created with.
///
/// On a desktop this is a window-system handle — the thing a compositor asks for its surfaces.
/// Here the display *is* the panel: its pixels are registered with [`panel::set`], and the frame
/// buffer is the panel's own. So this type carries nothing, and answers the two
/// `raw-window-handle` questions with [`HandleError::Unavailable`] rather than pretending: nothing
/// on this board has a handle to give, and a caller that needs one should hear that rather than
/// receive a made-up pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Panel;

impl HasWindowHandle for Panel {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

impl HasDisplayHandle for Panel {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, HandleError> {
        Err(HandleError::Unavailable)
    }
}

/// iced's compositor, over the panel's frame buffer.
pub struct Compositor {
    settings: Settings,
    sink: Sink,
}

impl iced_graphics::compositor::Compositor for Compositor {
    type Renderer = Renderer;
    type Surface = Surface;

    async fn with_backend(
        settings: Settings,
        _display: impl iced_graphics::compositor::Display,
        _compatible_window: impl iced_graphics::compositor::Window,
        _shell: Shell,
        backend: Option<&str>,
    ) -> Result<Self, Error> {
        match backend {
            None | Some("pomelo-gfx" | "pomelo_gfx") => {
                let sink = panel::take().ok_or(Error::GraphicsAdapterNotFound {
                    backend: "pomelo-gfx",
                    reason: Reason::RequestFailed(String::from(
                        "no panel was registered: `iced_pomelo_gfx::panel::set` has to run before \
                         a compositor is created",
                    )),
                })?;

                Ok(Self { settings, sink })
            }
            Some(backend) => Err(Error::GraphicsAdapterNotFound {
                backend: "pomelo-gfx",
                reason: Reason::DidNotMatch {
                    preferred_backend: String::from(backend),
                },
            }),
        }
    }

    fn create_renderer(&self) -> Self::Renderer {
        Renderer::new(self.settings.default_font, self.settings.default_text_size)
    }

    /// Allocates a panel-sized frame buffer.
    ///
    /// iced's contract has no way to report a failure here, and the honest answer to "the panel's
    /// pixels could not be allocated" is not a zero-sized surface that silently draws nowhere: it
    /// is a panic naming the size. (The shell can still fail *before* this, when no panel was
    /// registered at all — that one is a `Result`.)
    fn create_surface<W: iced_graphics::compositor::Window + Clone>(
        &mut self,
        _window: W,
        width: u32,
        height: u32,
    ) -> Self::Surface {
        panel_surface(width, height)
    }

    /// Rebuilds the buffers at a new size. The panel's own size never changes; this is here because
    /// the contract asks for it, and the recorded scene starts empty, so the next frame repaints
    /// everything.
    fn configure_surface(&mut self, surface: &mut Self::Surface, width: u32, height: u32) {
        *surface = panel_surface(width, height);
    }

    fn information(&self) -> Information {
        Information {
            adapter: String::from("RGB565 panel"),
            backend: String::from("pomelo-gfx"),
        }
    }

    fn present(
        &mut self,
        renderer: &mut Self::Renderer,
        surface: &mut Self::Surface,
        _viewport: &Viewport,
        background_color: Color,
        on_pre_present: impl FnOnce(),
    ) -> Result<(), SurfaceError> {
        let damage = surface.present(renderer, background_color);

        on_pre_present();

        // Nothing changed, so the panel is not touched at all: the loop's idle frames end here,
        // and on this hardware that is the difference between a still screen and a full QSPI
        // transfer per frame.
        if !damage.is_empty() {
            self.sink.present(surface.panel().data(), &damage);
        }

        Ok(())
    }

    /// Draws the recording into a fresh buffer and hands it back as RGBA8888 — the renderer's own
    /// [`Headless`](iced_core::renderer::Headless) screenshot, which needs no surface and no
    /// window.
    fn screenshot(
        &mut self,
        renderer: &mut Self::Renderer,
        viewport: &Viewport,
        background_color: Color,
    ) -> Vec<u8> {
        iced_core::renderer::Headless::screenshot(
            renderer,
            viewport.physical_size(),
            viewport.scale_factor(),
            background_color,
        )
    }
}

/// The compositor this renderer reports, and the one iced builds when it runs a `Program`.
impl iced_graphics::compositor::Default for Renderer {
    type Compositor = Compositor;
}

fn panel_surface(width: u32, height: u32) -> Surface {
    Surface::new(width, height).unwrap_or_else(|| {
        panic!(
            "a {width}x{height} panel needs {} KiB for its frame buffer",
            (width as usize * height as usize * 2) / 1024
        )
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::future::Future;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    use iced_core::renderer::Quad;
    use iced_core::renderer::Renderer as _;
    use iced_core::{Background, Font, Pixels, Point, Rectangle, Size};
    use iced_graphics::compositor::Compositor as _;
    use pomelo_gfx::rgb565_to_rgb888;

    use super::*;

    const SIZE: u32 = 32;

    /// What a panel was handed: the whole frame buffer, and the rectangles that changed.
    type Presented = Rc<RefCell<Vec<(Vec<u16>, Vec<Rectangle>)>>>;

    /// Registers a panel that keeps what it is given.
    fn recording_panel() -> Presented {
        let presented = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&presented);

        panel::set(move |pixels, damage| {
            sink.borrow_mut().push((pixels.to_vec(), damage.to_vec()));
        });

        presented
    }

    /// Polls a future to completion.
    ///
    /// The future `with_backend` returns is ready on its first poll — there is no I/O behind a
    /// panel — so this is a loop and not a runtime. `Waker::noop` is what keeps it four lines.
    fn block_on<F: Future>(future: F) -> F::Output {
        let mut future = Box::pin(future);
        let mut context = Context::from_waker(Waker::noop());

        loop {
            if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    /// A compositor over the panel that was just registered.
    fn compositor() -> Compositor {
        block_on(<Compositor as iced_graphics::compositor::Compositor>::new(
            Settings::default(),
            Panel,
            Panel,
            Shell::headless(),
        ))
        .expect("a registered panel is a display this compositor can use")
    }

    /// A renderer with a white square in the middle of the panel.
    fn frame() -> Renderer {
        let mut renderer = Renderer::new(Font::default(), Pixels(16.0));

        renderer.reset(Rectangle::with_size(Size::new(SIZE as f32, SIZE as f32)));
        renderer.fill_quad(
            Quad {
                bounds: Rectangle::new(Point::new(8.0, 8.0), Size::new(16.0, 16.0)),
                ..Quad::default()
            },
            Background::Color(Color::WHITE),
        );

        renderer
    }

    #[test]
    fn a_changed_frame_reaches_the_panel_and_an_identical_one_does_not() {
        let presented = recording_panel();
        let mut compositor = compositor();
        let mut surface = compositor.create_surface(Panel, SIZE, SIZE);
        let mut renderer = frame();
        let viewport = surface.viewport();

        compositor
            .present(&mut renderer, &mut surface, &viewport, Color::BLACK, || {})
            .expect("a panel always accepts the pixels");

        {
            let presented = presented.borrow();
            assert_eq!(presented.len(), 1, "the first frame is presented");

            let (pixels, damage) = &presented[0];
            assert_eq!(
                pixels.len(),
                (SIZE * SIZE) as usize,
                "the whole frame buffer is handed over, not the damaged rows"
            );
            assert!(!damage.is_empty(), "the first frame damages something");
            assert_eq!(
                rgb565_to_rgb888(pixels[(16 * SIZE + 16) as usize]),
                (255, 255, 255),
                "the quad is in the pixels"
            );
        }

        // The same recording again. Nothing changed, so the panel is not touched at all -- which
        // on this hardware is a QSPI transfer and a flush saved, every idle frame.
        let mut renderer = frame();

        compositor
            .present(&mut renderer, &mut surface, &viewport, Color::BLACK, || {})
            .expect("presenting an unchanged frame is not an error");

        assert_eq!(
            presented.borrow().len(),
            1,
            "an identical frame is not presented"
        );
    }

    #[test]
    fn there_is_one_panel_and_the_first_compositor_takes_it() {
        recording_panel();

        let _first = compositor();

        let second = block_on(<Compositor as iced_graphics::compositor::Compositor>::new(
            Settings::default(),
            Panel,
            Panel,
            Shell::headless(),
        ));

        assert!(
            matches!(second, Err(Error::GraphicsAdapterNotFound { .. })),
            "one panel cannot be presented to twice"
        );
    }

    #[test]
    fn a_compositor_without_a_panel_says_so() {
        let result = block_on(<Compositor as iced_graphics::compositor::Compositor>::new(
            Settings::default(),
            Panel,
            Panel,
            Shell::headless(),
        ));

        assert!(
            matches!(result, Err(Error::GraphicsAdapterNotFound { .. })),
            "the error names the missing registration rather than drawing nowhere"
        );
    }

    #[test]
    fn a_backend_that_is_not_this_one_leaves_the_panel_alone() {
        let presented = recording_panel();

        let wrong = block_on(
            <Compositor as iced_graphics::compositor::Compositor>::with_backend(
                Settings::default(),
                Panel,
                Panel,
                Shell::headless(),
                Some("wgpu"),
            ),
        );

        assert!(
            matches!(wrong, Err(Error::GraphicsAdapterNotFound { .. })),
            "the renderer that draws here is asked for by name or not at all"
        );

        // And the refusal did not consume the registration: a compositor for *this* backend still
        // finds its panel.
        let mut compositor = compositor();
        let mut surface = compositor.create_surface(Panel, SIZE, SIZE);
        let viewport = surface.viewport();
        let mut renderer = frame();

        compositor
            .present(&mut renderer, &mut surface, &viewport, Color::BLACK, || {})
            .expect("present");

        assert_eq!(presented.borrow().len(), 1);
    }

    #[test]
    fn a_screenshot_needs_no_surface() {
        recording_panel();

        let mut compositor = compositor();
        let mut renderer = frame();

        let bytes = compositor.screenshot(
            &mut renderer,
            &Viewport::with_physical_size(Size::new(SIZE, SIZE), 1.0),
            Color::BLACK,
        );

        assert_eq!(
            bytes.len(),
            (SIZE * SIZE * 4) as usize,
            "RGBA for every pixel"
        );
        assert_eq!(
            &bytes[(16 * SIZE + 16) as usize * 4..][..3],
            &[255, 255, 255],
            "the quad is in the screenshot"
        );
    }

    #[test]
    fn a_fresh_renderer_has_drawn_nothing() {
        recording_panel();

        let compositor = compositor();
        let mut renderer = compositor.create_renderer();

        assert!(renderer.layers()[0].quads.is_empty());
        assert_eq!(compositor.information().backend, "pomelo-gfx");
    }
}
