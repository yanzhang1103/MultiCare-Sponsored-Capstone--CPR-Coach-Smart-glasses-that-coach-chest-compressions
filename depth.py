"""Compression depth from head travel.

The idea being tested: with arms locked, the head moves down and up
about as far as the chest does. So head travel per compression, times
a calibration, estimates chest depth.

Two steps:
1. head_travel: how far the head moved on each compression.
2. A Calibration (fitted against the manikin's depth) turns head
   travel into chest depth.

Whether one calibration works for everyone is the open question. The
functions here let that be tested the same way on simulated and real
sessions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .rate import RESAMPLE_HZ

LOWPASS_HZ = 6.0            # smooths sensor noise before measuring travel
MAX_HALF_CYCLE_S = 0.75     # how far either side of the bottom to look for the top
BOTTOM_WINDOW_S = 0.10      # how far either side of the detected time to look for the bottom


def head_travel(
    t,
    y,
    compression_times,
    resample_hz: float = RESAMPLE_HZ,
    lowpass_hz: float = LOWPASS_HZ,
    max_half_cycle_s: float = MAX_HALF_CYCLE_S,
    bottom_window_s: float = BOTTOM_WINDOW_S,
) -> np.ndarray:
    """Head travel in metres for each detected compression.

    t, y: head height samples (seconds, metres).
    compression_times: bottom-of-compression times from detect_compressions.

    For each compression: find the lowest point near the detected time,
    and the highest point before it and after it (no further back or
    forward than the neighbouring compression). Travel is the average
    of the two drops. Averaging both sides cancels slow posture sway.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    times = np.asarray(compression_times, dtype=float)
    if times.size == 0:
        return np.array([])

    tu = np.arange(t[0], t[-1], 1.0 / resample_hz)
    yu = np.interp(tu, t, y)
    sos = butter(2, lowpass_hz, btype="lowpass", fs=resample_hz, output="sos")
    ys = sosfiltfilt(sos, yu)

    def window(a: float, b: float) -> np.ndarray:
        i0 = int(np.searchsorted(tu, a, side="left"))
        i1 = int(np.searchsorted(tu, b, side="right"))
        return ys[i0:i1]

    travel = np.full(times.size, np.nan)
    for k, tc in enumerate(times):
        bottom_w = window(tc - bottom_window_s, tc + bottom_window_s)
        if bottom_w.size == 0:
            continue
        bottom = bottom_w.min()

        before_start = tc - max_half_cycle_s
        if k > 0:
            before_start = max(before_start, times[k - 1])
        after_end = tc + max_half_cycle_s
        if k < times.size - 1:
            after_end = min(after_end, times[k + 1])

        drops = []
        for w in (window(before_start, tc), window(tc, after_end)):
            if w.size:
                drops.append(w.max() - bottom)
        if drops:
            travel[k] = float(np.mean(drops))
    return travel


@dataclass
class Calibration:
    """Chest depth = gain x head travel + offset."""

    gain: float
    offset_m: float = 0.0

    def depth(self, travel_m) -> np.ndarray:
        return self.gain * np.asarray(travel_m, dtype=float) + self.offset_m


def fit_calibration(
    travel_m,
    true_depth_m,
    through_origin: bool = False,
    trim_outliers: bool = True,
) -> Calibration:
    """Fit a calibration from paired head travel and manikin depth.

    through_origin: fit a gain only (no offset).
    trim_outliers:  fit once, drop points far from the line, fit again.
                    Keeps a few glances or mis-paired compressions from
                    bending the fit.
    The data needs a spread of depths (shallow to deep). A session at a
    single depth cannot pin down the line.
    """
    x = np.asarray(travel_m, dtype=float)
    d = np.asarray(true_depth_m, dtype=float)
    ok = ~np.isnan(x) & ~np.isnan(d)
    x, d = x[ok], d[ok]
    if x.size < 5:
        raise ValueError("need at least 5 paired compressions to fit a calibration")

    cal = _fit_line(x, d, through_origin)
    if trim_outliers:
        resid = d - cal.depth(x)
        spread = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        keep = np.abs(resid - np.median(resid)) <= 3.0 * max(spread, 1e-6)
        if 5 <= np.count_nonzero(keep) < x.size:
            cal = _fit_line(x[keep], d[keep], through_origin)
    return cal


def _fit_line(x: np.ndarray, d: np.ndarray, through_origin: bool) -> Calibration:
    if through_origin:
        return Calibration(gain=float(np.dot(x, d) / np.dot(x, x)), offset_m=0.0)
    gain, offset = np.polyfit(x, d, 1)
    return Calibration(gain=float(gain), offset_m=float(offset))
