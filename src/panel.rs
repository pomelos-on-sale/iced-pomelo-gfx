//! Where presented pixels go.
//!
//! iced's compositor is created with a *display connection* — a window handle on a desktop — and
//! that is how it knows what to draw into. There is no window here: the panel is a QSPI
//! peripheral, and iced's `Compositor::new(settings, display, window, shell)` has nowhere to put
//! it. So it is registered instead, exactly as the platform layer registers its board for
//! `run(program)`, and for the same reason: the entry point iced offers takes no hardware.
//!
//! One slot, taken when a compositor is created. A second compositor in the same process
//! therefore finds nothing to present to, which is the truth of this board: there is one panel.

use std::cell::RefCell;

use iced_core::Rectangle;

/// What a panel is handed once per changed frame: the whole RGB565 frame buffer, and the
/// rectangles that changed, in physical pixels.
type Presenter = Box<dyn Fn(&[u16], &[Rectangle])>;

/// The panel a compositor presents to: somewhere to put pixels, and no way to ask it anything.
pub(crate) struct Sink {
    present: Presenter,
}

impl Sink {
    /// Hands one frame to the panel: the whole RGB565 buffer, and the rectangles that changed.
    ///
    /// The buffer is passed whole rather than as the damaged rows because that is what the
    /// hardware takes — `host_lcd_draw_bitmap` walks the rows itself, with the frame buffer's
    /// origin, so that a full-width buffer needs no copy per rectangle.
    pub(crate) fn present(&self, pixels: &[u16], damage: &[Rectangle]) {
        (self.present)(pixels, damage);
    }
}

thread_local! {
    /// The panel the compositor will take, until it does.
    static PANEL: RefCell<Option<Sink>> = const { RefCell::new(None) };
}

/// Hands the compositor somewhere to put pixels.
///
/// `present` is called once per frame that changed anything, with the panel's whole RGB565 frame
/// buffer and the rectangles that changed, in physical pixels. Call this before creating a
/// compositor; the registration is taken by the first one.
pub fn set(present: impl Fn(&[u16], &[Rectangle]) + 'static) {
    PANEL.with(|slot| {
        *slot.borrow_mut() = Some(Sink {
            present: Box::new(present),
        })
    });
}

/// Takes the registered panel, leaving the slot empty.
pub(crate) fn take() -> Option<Sink> {
    PANEL.with(|slot| slot.borrow_mut().take())
}
