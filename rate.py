"""Compression rate and pause detection from head height.

Input: the glasses' head height over time (metres, seconds), taken from
the 6DoF head pose. The head rises and falls once per compression, so
each compression shows up as one dip in the signal.

Everything here is plain numbers in, numbers out, so it can be scored
on recorded sessions and later translated to C# for the live app.

Note for the live port: this version filters the whole recording at
once (zero-phase). The live version needs a causal filter, which adds
a small delay. The peak rules stay the same.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

# Defaults. All are tunable once real recordings exist.
RESAMPLE_HZ = 50.0          # uniform rate the signal is resampled to
BAND_HZ = (0.8, 4.0)        # keeps 48 to 240 compressions per minute
MIN_TRAVEL_M = 0.008        # ignore dips smaller than 8 mm of head travel
REL_PROMINENCE = 0.4        # ignore dips under 40% of the typical dip
MAX_RATE_CPM = 220.0        # no two compressions closer than this allows
ISOLATED_GAP_S = 1.5        # dips further apart than this are not in the same run
MIN_RUN = 4                 # fewer dips in a row than this is not CPR
PAUSE_THRESHOLD_S = 2.0     # a gap this long between compressions is a pause


def detect_compressions(
    t,
    y,
    resample_hz: float = RESAMPLE_HZ,
    band_hz: tuple[float, float] = BAND_HZ,
    min_travel_m: float = MIN_TRAVEL_M,
    rel_prominence: float = REL_PROMINENCE,
    max_rate_cpm: float = MAX_RATE_CPM,
    isolated_gap_s: float = ISOLATED_GAP_S,
    min_run: int = MIN_RUN,
) -> np.ndarray:
    """Find the time of the bottom of each compression.

    t: sample times in seconds (need not be evenly spaced).
    y: head height in metres at those times.
    Returns the times, in seconds, of each detected compression.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size != y.size:
        raise ValueError("t and y must be the same length")
    if t.size < 2 or (t[-1] - t[0]) < 2.0:
        return np.array([])

    # 1. Resample to an even time grid.
    tu = np.arange(t[0], t[-1], 1.0 / resample_hz)
    yu = np.interp(tu, t, y)

    # 2. Band-pass: removes slow posture sway and fast sensor noise.
    sos = butter(2, band_hz, btype="bandpass", fs=resample_hz, output="sos")
    yf = sosfiltfilt(sos, yu)

    # 3. Candidate dips: far enough apart and deep enough in absolute terms.
    min_distance = max(1, int(round(resample_hz * 60.0 / max_rate_cpm)))
    idx, props = find_peaks(-yf, distance=min_distance, prominence=min_travel_m)
    if idx.size == 0:
        return np.array([])

    # 4. Keep dips that are a fair fraction of the typical dip, so small
    #    wobbles between real compressions are not counted.
    typical = np.median(props["prominences"])
    idx = idx[props["prominences"] >= rel_prominence * typical]

    # 5. Refine each time to between samples (parabola through 3 points).
    times = np.array([_refine(tu, -yf, i) for i in idx])

    # 6. Keep only runs of compressions. A lone dip or two is a glance or
    #    a shift of posture, not CPR.
    return _keep_runs(times, isolated_gap_s, min_run)


def run_mask(times, max_gap_s: float = ISOLATED_GAP_S, min_run: int = MIN_RUN) -> np.ndarray:
    """True for each compression that sits in a run of at least min_run.

    A run is a stretch where each compression follows the last within
    max_gap_s. Short runs (a glance, or the three-compression start cue)
    come out False.
    """
    times = np.asarray(times, dtype=float)
    mask = np.zeros(times.size, dtype=bool)
    if times.size == 0:
        return mask
    breaks = np.flatnonzero(np.diff(times) > max_gap_s) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [times.size]))
    for a, b in zip(starts, ends):
        if b - a >= min_run:
            mask[a:b] = True
    return mask


def _keep_runs(times: np.ndarray, max_gap_s: float, min_run: int) -> np.ndarray:
    return times[run_mask(times, max_gap_s, min_run)]


def _refine(tu: np.ndarray, v: np.ndarray, i: int) -> float:
    """Sub-sample peak time from a parabola through the peak and its neighbours."""
    if i <= 0 or i >= v.size - 1:
        return float(tu[i])
    a, b, c = v[i - 1], v[i], v[i + 1]
    denom = a - 2.0 * b + c
    if denom == 0:
        return float(tu[i])
    shift = 0.5 * (a - c) / denom
    shift = float(np.clip(shift, -0.5, 0.5))
    return float(tu[i] + shift * (tu[1] - tu[0]))


def rate_in_window(
    times,
    t0: float,
    t1: float,
    pause_threshold_s: float = PAUSE_THRESHOLD_S,
    min_intervals: int = 3,
) -> float:
    """Compression rate (per minute) between t0 and t1.

    The rate is 60 divided by the middle (median) gap between
    compressions. The median is used so that one missed or extra
    compression does not swing the rate.
    Gaps long enough to be a pause are left out, so a pause does not
    drag the rate down; it is reported separately by find_pauses.
    Returns NaN when there are too few compressions to say.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return float("nan")
    gaps = np.diff(times)
    mids = (times[:-1] + times[1:]) / 2.0
    use = (mids >= t0) & (mids < t1) & (gaps <= pause_threshold_s)
    if np.count_nonzero(use) < min_intervals:
        return float("nan")
    return float(60.0 / np.median(gaps[use]))


def rate_series(
    times,
    t_eval,
    window_s: float = 5.0,
    pause_threshold_s: float = PAUSE_THRESHOLD_S,
) -> np.ndarray:
    """Rate at each moment in t_eval, using the window_s seconds before it.

    This is what the live app would show: a rolling rate.
    """
    return np.array(
        [rate_in_window(times, te - window_s, te, pause_threshold_s) for te in t_eval]
    )


def find_pauses(
    times, pause_threshold_s: float = PAUSE_THRESHOLD_S
) -> list[tuple[float, float]]:
    """Pauses as (start, end): gaps between compressions over the threshold."""
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return []
    gaps = np.diff(times)
    return [
        (float(times[i]), float(times[i + 1]))
        for i in np.flatnonzero(gaps > pause_threshold_s)
    ]
