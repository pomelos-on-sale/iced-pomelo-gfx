//! What a frame spends, phase by phase.
//!
//! A screen change on the panel costs hundreds of milliseconds, and "the frame is slow" is not
//! actionable: this module is what turns it into "these are the milliseconds, in this order".
//! Every phase of a frame is timed with one `Instant` pair, the totals are collected in counters,
//! and the compositor prints them as one line when it has a frame to hand the panel.
//!
//! It is compiled only under the `profile` feature. Without it every function here is a no-op —
//! `start` hands back a guard that holds nothing and dropping it does nothing — so a call site is
//! the same plain statement in both builds and there is no `#[cfg]` in the middle of a render loop.
//! A build without the feature must not read the clock twice per glyph for numbers nobody reads.
//!
//! # What the split is for
//!
//! Measured on the launcher opening its settings screen (a whole-panel frame, x86 release): the
//! tree is ~12% of it, text ~60%, and the whole-screen `clear` ~7%. Of the text, most is
//! **rasterising a glyph that was not in the cache yet** — a screen pays that once (the same screen
//! a second time is 3× cheaper) and pays only the blits after that. Rasterising, the blit, the
//! `clear` and the panel write are all per-glyph or per-pixel costs, which is what scales with the
//! hardware; the raw "paint" number in the firmware's own log cannot say which of them it is.

use std::time::Duration;

/// A phase of a frame, as deep as the split needs to go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// `UserInterface::build`: the view function, layout, and the widget diff that follows.
    TreeBuild,
    /// `UserInterface::update`: the events, which is where a message lands.
    TreeUpdate,
    /// `UserInterface::draw`: the widgets recording their commands into the layers.
    TreeRecord,
    /// The damage bookkeeping: `diff`, `group`, and this crate's own fold of the result.
    Damage,
    /// Setting the clip and clearing the damaged rectangle back to the background.
    Clear,
    /// Replaying the recorded quads and canvas primitives.
    Quads,
    /// Replaying the recorded text.
    Text,
    /// Rasterising a glyph that was not in the cache yet.
    Rasterise,
    /// Blitting one glyph's cached mask.
    Blits,
}

impl Phase {
    /// How many phases a frame is split into.
    pub const COUNT: usize = 9;

    /// Every phase, in the order a frame goes through them.
    pub const ALL: [Phase; Self::COUNT] = [
        Phase::TreeBuild,
        Phase::TreeUpdate,
        Phase::TreeRecord,
        Phase::Damage,
        Phase::Clear,
        Phase::Quads,
        Phase::Text,
        Phase::Rasterise,
        Phase::Blits,
    ];

    /// The word this phase prints under.
    pub const fn name(self) -> &'static str {
        match self {
            Phase::TreeBuild => "build",
            Phase::TreeUpdate => "update",
            Phase::TreeRecord => "record",
            Phase::Damage => "damage",
            Phase::Clear => "clear",
            Phase::Quads => "quads",
            Phase::Text => "text",
            Phase::Rasterise => "raster",
            Phase::Blits => "blit",
        }
    }

    /// Whether this phase happens *inside* another one.
    ///
    /// Both of the text ones do: a blit is drawn within the text loop and a rasterise within a
    /// blit's mask lookup, which is the split's whole point — the frame's own cost is the outer
    /// phases, and a text-heavy screen spends most of its `text` in `raster`.
    pub const fn is_nested(self) -> bool {
        matches!(self, Phase::Rasterise | Phase::Blits)
    }
}

/// One frame's numbers: nanoseconds per phase, and the glyph counts that explain the text ones.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// The time each phase took, indexed by [`Phase`] as `usize`.
    pub phases: [u64; Phase::COUNT],
    /// Glyphs whose mask was blitted in this frame.
    pub glyphs: u64,
    /// How many of them had to be rasterised first — the part a screen pays only once.
    pub rasterised: u64,
}

impl Frame {
    /// The time one phase of the frame took.
    pub fn phase(&self, phase: Phase) -> Duration {
        Duration::from_nanos(self.phases[phase as usize])
    }

    /// Everything the split accounts for: the outermost phases, so the two that happen inside
    /// `text` are counted there and not again here.
    ///
    /// It is the frame's own cost minus whatever is around it — the panel write, most of all,
    /// which belongs to the host and not to the renderer, and this build's own printing.
    pub fn total(&self) -> Duration {
        Duration::from_nanos(
            self.phases
                .iter()
                .zip(Phase::ALL)
                .filter(|(_, phase)| !phase.is_nested())
                .map(|(nanos, _)| nanos)
                .sum(),
        )
    }

    /// Whether the frame did nothing at all, which is what an idle frame looks like.
    pub fn is_empty(&self) -> bool {
        self.phases.iter().all(|nanos| *nanos == 0) && self.glyphs == 0
    }
}
/// Times `phase` until the guard is dropped.
///
/// The guard is the whole interface on purpose: a phase that ends early — a `?`, a `return`, a
/// `continue` inside it — still gets counted, and a phase that is not taken (a layer with no text
/// in it) costs nothing beyond the drop.
#[must_use = "the guard is what records the phase; dropping it immediately records nothing"]
pub fn start(phase: Phase) -> Timer {
    Timer::new(phase)
}

/// One phase being timed, or nothing at all when the crate is built without `profile`.
#[derive(Debug)]
pub struct Timer {
    #[cfg(feature = "profile")]
    phase: Phase,
    #[cfg(feature = "profile")]
    started: std::time::Instant,
}

impl Timer {
    fn new(phase: Phase) -> Self {
        #[cfg(feature = "profile")]
        {
            Self {
                phase,
                started: std::time::Instant::now(),
            }
        }

        #[cfg(not(feature = "profile"))]
        {
            let _ = phase;

            Self {}
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        #[cfg(feature = "profile")]
        counters::add(self.phase, self.started.elapsed());
    }
}

/// Counts one blitted glyph, saying whether it had to be rasterised first.
pub fn glyph(rasterised: bool) {
    #[cfg(feature = "profile")]
    counters::glyph(rasterised);

    #[cfg(not(feature = "profile"))]
    let _ = rasterised;
}

/// Takes the counters of the frame that has just been drawn, leaving them at zero.
pub fn take() -> Frame {
    #[cfg(feature = "profile")]
    return counters::take();

    #[cfg(not(feature = "profile"))]
    Frame::default()
}

/// Prints the frame that has just been drawn, if it drew anything at all.
///
/// One line, because this goes to a serial console: the phases in milliseconds, the glyph counts,
/// and the sum of the phases so that a phase nothing measured would stand out as a gap against
/// the host's own frame time.
pub fn report() {
    #[cfg(feature = "profile")]
    {
        let frame = take();

        if frame.is_empty() {
            return;
        }

        let mut line = String::from("[phase]");

        for phase in Phase::ALL {
            line.push_str(&format!(
                " {} {:.2}",
                phase.name(),
                frame.phase(phase).as_secs_f64() * 1000.0
            ));
        }

        println!(
            "{line}  glyphs {}/{}  total {:.2}ms",
            frame.glyphs,
            frame.rasterised,
            frame.total().as_secs_f64() * 1000.0
        );
    }
}

/// The counters themselves: atomics, because a `canvas` widget can draw from a task and the
/// compositor reads them from the loop.
///
/// They are `u32` nanoseconds and not `u64`, because the target this exists for is 32-bit and has
/// no 64-bit atomics at all (the firmware will not compile without this). A phase that ran longer
/// than four seconds would not fit in one, which is why the addition saturates: a number that is
/// wrong in the direction of "small" would read as a fast frame.
#[cfg(feature = "profile")]
mod counters {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    use super::{Frame, Phase};

    static PHASES: [AtomicU32; Phase::COUNT] = [const { AtomicU32::new(0) }; Phase::COUNT];
    static GLYPHS: AtomicU32 = AtomicU32::new(0);
    static RASTERISED: AtomicU32 = AtomicU32::new(0);

    pub(super) fn add(phase: Phase, spent: Duration) {
        let nanos = u32::try_from(spent.as_nanos()).unwrap_or(u32::MAX);

        let _ =
            PHASES[phase as usize].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
                Some(total.saturating_add(nanos))
            });
    }

    pub(super) fn glyph(rasterised: bool) {
        GLYPHS.fetch_add(1, Ordering::Relaxed);

        if rasterised {
            RASTERISED.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn take() -> Frame {
        let mut phases = [0; Phase::COUNT];

        for (slot, counter) in phases.iter_mut().zip(&PHASES) {
            *slot = u64::from(counter.swap(0, Ordering::Relaxed));
        }

        Frame {
            phases,
            glyphs: u64::from(GLYPHS.swap(0, Ordering::Relaxed)),
            rasterised: u64::from(RASTERISED.swap(0, Ordering::Relaxed)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_phase_has_a_name() {
        for phase in Phase::ALL {
            assert!(!phase.name().is_empty());
        }
    }

    #[test]
    fn an_idle_frame_is_empty_and_a_measured_one_is_not() {
        assert!(Frame::default().is_empty());

        let mut frame = Frame::default();
        frame.phases[Phase::Quads as usize] = 1_000;

        assert!(!frame.is_empty());
        assert_eq!(frame.phase(Phase::Quads), Duration::from_micros(1));
        assert_eq!(frame.total(), Duration::from_micros(1));
    }

    /// A blit and a rasterise are drawn *inside* the text loop, so a screen that spends 5 ms in
    /// text and 3 of them in rasterising has spent 5 ms on text and not 8. This is the arithmetic
    /// the device's own numbers depend on: the first version of this line added every phase up and
    /// reported a 618 ms frame inside a 412 ms one, which is the kind of number that sends someone
    /// looking for a bug in the wrong crate.
    #[test]
    fn the_nested_phases_are_not_counted_twice() {
        let mut frame = Frame::default();
        frame.phases[Phase::Text as usize] = 5_000;
        frame.phases[Phase::Rasterise as usize] = 3_000;
        frame.phases[Phase::Blits as usize] = 500;

        assert_eq!(frame.total(), Duration::from_micros(5));
        assert_eq!(frame.phase(Phase::Rasterise), Duration::from_micros(3));
    }

    /// Without the feature the whole module is inert, which is what keeps it out of a build that
    /// does not want it: nothing may be recorded, not even by accident.
    #[test]
    fn a_timer_records_nothing_without_the_feature() {
        let guard = start(Phase::Text);
        drop(guard);

        glyph(true);

        #[cfg(not(feature = "profile"))]
        assert!(take().is_empty());

        #[cfg(feature = "profile")]
        assert_eq!(take().glyphs, 1);
    }
}
