"""Fatigue: notice when compressions fade against the person's own start.

People tire within a minute or two of CPR and usually do not notice:
their compressions get shallower while feeling the same. This watches
how far the head travels on each compression and compares the recent
ones with the same person's first 30 seconds (decision log E6).

It needs no depth calibration, because it compares a person with
themselves. So it works on a pillow as well as on a manikin.

Two outputs:
    tiring     head travel has dropped by a set share and stayed there
    swap due   two minutes of compressions have gone by (the guideline
               interval for changing rescuers), whatever the readings say

A rest of 30 seconds or more starts everything again: fresh baseline,
fresh two-minute clock.

Plain arithmetic and short lists, to translate to C# line for line.
"""

from __future__ import annotations

from dataclasses import dataclass

NAN = float("nan")


@dataclass
class FatigueConfig:
    baseline_s: float = 30.0        # the person's own first stretch of compressions
    baseline_min_n: int = 20        # ...and at least this many compressions in it
    recent_n: int = 20              # "recent" = the median of this many compressions
    decline: float = 0.15           # tiring when travel is down by this share
    recover: float = 0.07           # no longer tiring when it is back within this share
    sustain_n: int = 10             # the drop must hold for this many compressions in a row
    swap_after_s: float = 120.0     # remind to swap after this long (short pauses included)
    rest_reset_s: float = 30.0      # a rest this long starts everything again
    pause_s: float = 2.0            # gaps longer than this are not compression time


def _median(values) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class FatigueMonitor:
    def __init__(self, config: FatigueConfig | None = None):
        self.cfg = config or FatigueConfig()
        self._last_t: float | None = None
        self.reset()

    def reset(self) -> None:
        """Start again: after a long rest, or for a new person."""
        self.active_s = 0.0             # compression time since the start or the last rest
        self.elapsed_s = 0.0            # clock time since then, short pauses included
        self.baseline = NAN             # typical head travel at the start, metres
        self.decline = NAN              # share by which recent travel is below the baseline
        self.tiring = False
        self.swap_due = False
        self._baseline_from_s = 0.0     # compression time at which the baseline window began
        self._baseline_values: list[float] = []
        self._recent: list[float] = []
        self._n_down = 0
        self._n_back = 0
        self._swap_said_at_s = 0.0

    def rebaseline(self) -> None:
        """Measure the baseline afresh from now.

        Call this when the person has been told to ease off: getting
        shallower on purpose is not fatigue.
        """
        self.baseline = NAN
        self.decline = NAN
        self.tiring = False
        self._baseline_from_s = self.active_s
        self._baseline_values = []
        self._recent = []
        self._n_down = 0
        self._n_back = 0

    def swap_said(self) -> None:
        """The swap reminder has been given; start the next two minutes."""
        self.swap_due = False
        self._swap_said_at_s = self.elapsed_s

    def on_compression(self, t: float, travel_m: float) -> None:
        cfg = self.cfg
        if self._last_t is not None:
            gap = t - self._last_t
            if gap >= cfg.rest_reset_s:
                self.reset()
            elif gap > 0.0:
                self.elapsed_s += gap
                if gap <= cfg.pause_s:
                    self.active_s += gap
        self._last_t = t

        if self.elapsed_s - self._swap_said_at_s >= cfg.swap_after_s:
            self.swap_due = True

        if travel_m != travel_m:        # NaN: no reading for this one
            return

        if self.baseline != self.baseline:      # baseline not set yet
            self._baseline_values.append(travel_m)
            long_enough = self.active_s - self._baseline_from_s >= cfg.baseline_s
            if long_enough and len(self._baseline_values) >= cfg.baseline_min_n:
                self.baseline = _median(self._baseline_values)
            return

        self._recent.append(travel_m)
        if len(self._recent) > cfg.recent_n:
            self._recent.pop(0)
        if len(self._recent) < cfg.recent_n:
            return

        self.decline = 1.0 - _median(self._recent) / self.baseline
        if self.decline >= cfg.decline:
            self._n_down += 1
            self._n_back = 0
        elif self.decline <= cfg.recover:
            self._n_back += 1
            self._n_down = 0
        else:
            self._n_down = 0
            self._n_back = 0
        if self._n_down >= cfg.sustain_n:
            self.tiring = True
        if self._n_back >= cfg.sustain_n:
            self.tiring = False
