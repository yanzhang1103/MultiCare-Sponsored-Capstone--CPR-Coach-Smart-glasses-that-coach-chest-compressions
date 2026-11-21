"""Live compression detection: one head-pose sample in, compressions out.

This is the version that runs on the glasses. It does the same job as
rate.detect_compressions and depth.head_travel, but it only ever looks
backwards, so it can run while the person is compressing.

It is written to be translated to C# line for line: plain arithmetic,
fixed-size buffers, no array libraries in the per-sample path. The
filter numbers are computed once at start-up; in C# they become
constants (print them with StreamingDetector().filter_coefficients()).

Use:

    detector = StreamingDetector()
    for t, height in samples:                 # as each sample arrives
        for compression in detector.update(t, height):
            ...                               # usually none, sometimes one or more
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from scipy.signal import butter

from .rate import (
    BAND_HZ,
    ISOLATED_GAP_S,
    MAX_RATE_CPM,
    MIN_RUN,
    MIN_TRAVEL_M,
    REL_PROMINENCE,
    RESAMPLE_HZ,
)

LOWPASS_HZ = 6.0            # light smoothing of the signal that travel is measured on
CONFIRM_FRACTION = 0.15     # a dip is confirmed once the head has come back up this share of a typical dip
CONFIRM_FLOOR_M = 0.003     # ...and never by less than this
TYPICAL_OVER_N = 8          # "typical dip" = the median of this many recent dips
BUFFER_S = 2.5              # how much smoothed signal is kept for measuring travel
BOTTOM_LOOK_BACK_S = 0.15   # where to look for the true bottom around the filtered dip
BOTTOM_LOOK_AHEAD_S = 0.10  # ...which means waiting this long after the dip before measuring
MAX_HALF_CYCLE_S = 0.75     # how far back to look for the top before a dip
DROPOUT_S = 0.5             # a gap in the samples this long restarts the detector


@dataclass
class Compression:
    t: float              # time of the bottom of the compression (input clock)
    travel_m: float       # head travel on the way down, metres
    detected_at: float    # time the detector reported it; the difference is the delay
    prev_rise_m: float = float("nan")
    # prev_rise_m: how far the head came back up after the PREVIOUS compression.
    # Averaging it with that compression's travel_m cancels slow posture sway,
    # so a depth shown a moment later can be steadier than the first reading.


class _Biquad:
    """One second-order filter section (direct form II transposed)."""

    def __init__(self, b0, b1, b2, a1, a2):
        self.b0, self.b1, self.b2, self.a1, self.a2 = b0, b1, b2, a1, a2
        self.z1 = 0.0
        self.z2 = 0.0

    def step(self, x: float) -> float:
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y

    def reset(self) -> None:
        self.z1 = 0.0
        self.z2 = 0.0


def _sections(sos) -> list[_Biquad]:
    return [_Biquad(r[0], r[1], r[2], r[4], r[5]) for r in sos]


def _median(values) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class StreamingDetector:
    def __init__(
        self,
        rate_hz: float = RESAMPLE_HZ,
        band_hz: tuple[float, float] = BAND_HZ,
        lowpass_hz: float = LOWPASS_HZ,
        min_travel_m: float = MIN_TRAVEL_M,
        rel_prominence: float = REL_PROMINENCE,
        max_rate_cpm: float = MAX_RATE_CPM,
        run_gap_s: float = ISOLATED_GAP_S,
        min_run: int = MIN_RUN,
    ):
        self.dt = 1.0 / rate_hz
        self.min_travel_m = min_travel_m
        self.rel_prominence = rel_prominence
        self.min_interval_s = 60.0 / max_rate_cpm
        self.run_gap_s = run_gap_s
        self.min_run = min_run

        self._band_sos = butter(2, band_hz, btype="bandpass", fs=rate_hz, output="sos")
        self._low_sos = butter(2, lowpass_hz, btype="lowpass", fs=rate_hz, output="sos")
        self._band = _sections(self._band_sos)
        self._low = _sections(self._low_sos)
        # The smoothing filter reports everything this much late; times are corrected for it.
        self._low_delay_s = 0.225 / lowpass_hz

        self._buffer: deque[tuple[float, float]] = deque(maxlen=int(BUFFER_S * rate_hz))
        self._restart()

    # ---- public ---------------------------------------------------------------

    def update(self, t: float, height: float) -> list[Compression]:
        """Feed one head-pose sample. Returns the compressions it completes (often none)."""
        out: list[Compression] = []
        if self._prev_t is None or t - self._prev_t > DROPOUT_S or t < self._prev_t:
            self._restart()
            self._prev_t, self._prev_y = t, height
            self._next_tick = t
            self._origin = height
            return out

        # Step along an even time grid, taking the height on the straight
        # line between the last two samples.
        while self._next_tick <= t:
            span = t - self._prev_t
            w = (self._next_tick - self._prev_t) / span if span > 0 else 1.0
            y = self._prev_y + w * (height - self._prev_y)
            self._step(self._next_tick, y, out)
            self._next_tick += self.dt
        self._prev_t, self._prev_y = t, height
        return out

    def filter_coefficients(self) -> dict:
        """The filter numbers, for copying into the C# version."""
        return {
            "rate_hz": 1.0 / self.dt,
            "band_pass_sections_b0_b1_b2_a1_a2": [
                [float(r[0]), float(r[1]), float(r[2]), float(r[4]), float(r[5])]
                for r in self._band_sos
            ],
            "low_pass_sections_b0_b1_b2_a1_a2": [
                [float(r[0]), float(r[1]), float(r[2]), float(r[4]), float(r[5])]
                for r in self._low_sos
            ],
        }

    # ---- one grid step ----------------------------------------------------------

    def _restart(self) -> None:
        for f in self._band + self._low:
            f.reset()
        self._buffer.clear()
        self._prev_t = None
        self._prev_y = 0.0
        self._next_tick = 0.0
        self._origin = 0.0
        self._seeking_dip = True        # looking for a low point (else a high point)
        self._ext_v = 0.0               # most extreme filtered value so far in this search
        self._ext_t = 0.0
        self._last_top_v = 0.0
        self._pending_dip_t = None      # a dip waiting for its look-ahead before it is measured
        self._recent_travel: deque[float] = deque(maxlen=TYPICAL_OVER_N)
        self._last_bottom_t = None      # time and height of the last accepted bottom
        self._last_bottom_s = 0.0
        self._run: list[Compression] = []   # the current run of compressions
        self._run_reported = 0

    def _step(self, t: float, y: float, out: list[Compression]) -> None:
        # Filter the height relative to where it started, so the filters
        # do not ring at start-up.
        x = y - self._origin
        v = x
        for f in self._band:
            v = f.step(v)
        s = x
        for f in self._low:
            s = f.step(s)
        self._buffer.append((t, s + self._origin))

        typical = _median(self._recent_travel) if self._recent_travel else 0.0
        confirm = max(CONFIRM_FLOOR_M, CONFIRM_FRACTION * typical)

        # The band-passed signal proposes dips; each is then checked and
        # measured on the lightly smoothed signal a moment later.
        if self._seeking_dip:
            if v < self._ext_v:
                self._ext_v, self._ext_t = v, t
            elif v > self._ext_v + confirm:
                if self._last_top_v - self._ext_v >= 0.5 * self.min_travel_m:
                    self._pending_dip_t = self._ext_t
                self._seeking_dip = False
                self._ext_v, self._ext_t = v, t
        else:
            if v > self._ext_v:
                self._ext_v, self._ext_t = v, t
            elif v < self._ext_v - confirm:
                self._last_top_v = self._ext_v
                self._seeking_dip = True
                self._ext_v, self._ext_t = v, t

        if self._pending_dip_t is not None and t >= self._pending_dip_t + BOTTOM_LOOK_AHEAD_S:
            dip_t = self._pending_dip_t
            self._pending_dip_t = None
            self._on_dip(dip_t, typical, t, out)

    def _on_dip(self, dip_t: float, typical: float, now: float, out) -> None:
        bottom_t, bottom, top = self._measure(dip_t)
        if bottom is None or top is None:
            return
        travel = top - bottom
        # A real compression moves the head: by a minimum amount, and by a
        # fair share of what recent compressions did. This drops filter
        # ringing after the person stops, and small wobbles.
        if travel < max(self.min_travel_m, self.rel_prominence * typical):
            return
        if self._last_bottom_t is not None and bottom_t - self._last_bottom_t < self.min_interval_s:
            return

        prev_rise = float("nan")
        if self._last_bottom_t is not None and bottom_t - self._last_bottom_t <= MAX_HALF_CYCLE_S * 2:
            prev_rise = top - self._last_bottom_s
        self._last_bottom_t, self._last_bottom_s = bottom_t, bottom
        self._recent_travel.append(travel)

        comp = Compression(
            t=bottom_t - self._low_delay_s, travel_m=travel, detected_at=now, prev_rise_m=prev_rise
        )

        # Only runs of compressions count. The first few of a run are held
        # back until the run is long enough, then reported together.
        if self._run and comp.t - self._run[-1].t > self.run_gap_s:
            self._run = []
            self._run_reported = 0
        self._run.append(comp)
        if len(self._run) >= self.min_run:
            for held in self._run[self._run_reported:]:
                held.detected_at = now
                out.append(held)
            self._run_reported = len(self._run)
            if len(self._run) > 2 * self.min_run:      # keep the list short
                drop = len(self._run) - self.min_run
                self._run = self._run[drop:]
                self._run_reported -= drop

    def _measure(self, dip_t: float):
        """Bottom time, bottom height and the top before it, from the smoothed signal."""
        lo, hi = dip_t - BOTTOM_LOOK_BACK_S, dip_t + BOTTOM_LOOK_AHEAD_S
        bottom_t, bottom = dip_t, None
        for bt, bs in self._buffer:
            if lo <= bt <= hi and (bottom is None or bs < bottom):
                bottom_t, bottom = bt, bs
        if bottom is None:
            return dip_t, None, None

        start = bottom_t - MAX_HALF_CYCLE_S
        if self._last_bottom_t is not None and self._last_bottom_t > start:
            start = self._last_bottom_t
        top = None
        for bt, bs in self._buffer:
            if start <= bt <= bottom_t and (top is None or bs > top):
                top = bs
        return bottom_t, bottom, top
