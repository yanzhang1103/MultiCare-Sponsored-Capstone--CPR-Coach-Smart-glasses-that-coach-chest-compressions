"""Line up the glasses recording with the manikin recording.

The two devices keep their own clocks, so the same compression has a
different time stamp on each. This finds the mapping

    glasses time = manikin time + offset
                   + drift x (time since the manikin's first compression)

from the compressions themselves: the pattern of compressions, pauses
and the start cue is the same in both recordings, just shifted.

The risk is a steady rhythm: every compression looks like the next, so
the recordings can line up one compression out and still look right.
The start cue (a few slow, sharp compressions, then stillness) and
any pauses break that tie. The result reports how many more
compressions the best alignment matched than the runner-up, so a
doubtful alignment is flagged, not used quietly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, correlation_lags, find_peaks

from .rate import detect_compressions
from .scoring import pair_events

GRID_S = 0.01               # step of the coarse search
COARSE_SIGMA_S = 0.04       # blur of each compression in the coarse search
FINE_SIGMA_S = 0.02         # how tightly paired compressions must agree to score
PAIR_TOLERANCE_S = 0.15     # two time stamps this close can be the same compression
N_CANDIDATES = 9            # coarse matches that get a close look
MIN_PAIRS = 10              # fewer paired compressions than this is no alignment

# What counts as a trustworthy alignment.
OK_PAIRED_FRACTION = 0.80   # share of the manikin's compressions found in the glasses data
OK_RESIDUAL_MS = 30.0       # typical leftover timing difference
OK_MARGIN = 3.0             # compressions the best alignment must beat the runner-up by
OK_DRIFT_PPM = 1000.0       # clocks should not disagree by more than this


@dataclass
class Alignment:
    offset_s: float           # glasses clock minus manikin clock, at the manikin's first compression
    drift_ppm: float          # how much faster the glasses clock runs, parts per million
    ref_manikin_s: float      # manikin time of its first compression (where offset_s applies)
    n_paired: int             # compressions matched between the two recordings
    paired_fraction: float    # share of the manikin's compressions that were matched
    residual_ms: float        # typical leftover timing difference after alignment
    margin: float             # well-matched compressions: best alignment minus runner-up
    ok: bool                  # passes all four checks above
    reason: str               # why it is not ok, or "" if it is

    def to_glasses_time(self, manikin_time):
        """Convert manikin time stamps to the glasses clock."""
        m = np.asarray(manikin_time, dtype=float)
        d = self.drift_ppm * 1e-6
        return m + self.offset_s + d * (m - self.ref_manikin_s)

    def to_manikin_time(self, glasses_time):
        """Convert glasses time stamps to the manikin clock."""
        g = np.asarray(glasses_time, dtype=float)
        d = self.drift_ppm * 1e-6
        return (g - self.offset_s + d * self.ref_manikin_s) / (1.0 + d)


def align_session(
    t,
    y,
    manikin_times,
    fit_drift: bool = True,
    max_offset_s: float | None = None,
) -> Alignment:
    """Align from the raw glasses signal.

    Detects compressions with runs of three allowed, so a three-compression
    start cue is kept (the normal detector needs four in a row).
    """
    glasses_times = detect_compressions(t, y, min_run=3)
    return align_events(glasses_times, manikin_times, fit_drift, max_offset_s)


def align_events(
    glasses_times,
    manikin_times,
    fit_drift: bool = True,
    max_offset_s: float | None = None,
) -> Alignment:
    """Align two lists of compression times.

    max_offset_s: if the clocks are known to be within this many seconds
    of each other, say so; it narrows the search. None searches everything.
    """
    g = np.sort(np.asarray(glasses_times, dtype=float))
    m = np.sort(np.asarray(manikin_times, dtype=float))
    if g.size < MIN_PAIRS or m.size < MIN_PAIRS:
        return _failed("too few compressions in one of the recordings")

    # 1. Coarse search: slide one recording past the other and note the
    #    offsets where the compressions line up best.
    candidates = _coarse_offsets(g, m, max_offset_s)

    # 2. Close look at each: pair compressions, fine-tune, score by how
    #    tightly the pairs agree.
    results = []
    for tau in candidates:
        r = _evaluate(g, m, tau, fit_drift)
        if r is None:
            continue
        mid = 0.5 * (m[0] + m[-1])
        r["mid_shift"] = r["a"] * mid + r["b"] - mid
        if all(abs(r["mid_shift"] - o["mid_shift"]) > 0.05 for o in results):
            results.append(r)
    if not results:
        return _failed("the recordings do not line up anywhere")

    results.sort(key=lambda r: r["score"], reverse=True)
    best = results[0]
    margin = best["score"] - results[1]["score"] if len(results) > 1 else best["score"]

    drift_ppm = (best["a"] - 1.0) * 1e6
    mapped = best["a"] * m + best["b"]
    in_span = np.count_nonzero((mapped >= g[0] - 1.0) & (mapped <= g[-1] + 1.0))
    paired_fraction = best["n"] / in_span if in_span else 0.0
    residual_ms = best["resid_sd"] * 1000.0

    reasons = []
    if paired_fraction < OK_PAIRED_FRACTION:
        reasons.append(f"only {paired_fraction:.0%} of the manikin's compressions matched")
    if residual_ms > OK_RESIDUAL_MS:
        reasons.append(f"matched compressions disagree by {residual_ms:.0f} ms")
    if margin < OK_MARGIN:
        reasons.append("another alignment fits almost as well (steady rhythm, no clear cue)")
    if abs(drift_ppm) > OK_DRIFT_PPM:
        reasons.append(f"clock drift of {drift_ppm:.0f} ppm is implausibly large")

    ref = float(m[0])
    return Alignment(
        offset_s=float(best["a"] * ref + best["b"] - ref),
        drift_ppm=float(drift_ppm),
        ref_manikin_s=ref,
        n_paired=int(best["n"]),
        paired_fraction=float(paired_fraction),
        residual_ms=float(residual_ms),
        margin=float(margin),
        ok=not reasons,
        reason="; ".join(reasons),
    )


def _failed(reason: str) -> Alignment:
    nan = float("nan")
    return Alignment(nan, 0.0, nan, 0, 0.0, nan, 0.0, False, reason)


def _coarse_offsets(g: np.ndarray, m: np.ndarray, max_offset_s: float | None) -> list[float]:
    """Offsets (glasses minus manikin) where the two compression trains overlap best."""
    g0, m0 = g[0] - 1.0, m[0] - 1.0
    train_g = _train(g, g0)
    train_m = _train(m, m0)
    c = correlate(train_g, train_m, mode="full", method="fft")
    lags = correlation_lags(train_g.size, train_m.size, mode="full")
    tau = lags * GRID_S + (g0 - m0)
    if max_offset_s is not None:
        c = np.where(np.abs(tau) <= max_offset_s, c, 0.0)

    peaks, _ = find_peaks(c, distance=int(0.2 / GRID_S))
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(c))])
    top = peaks[np.argsort(c[peaks])[::-1][:N_CANDIDATES]]
    return [float(tau[i]) for i in top]


def _train(times: np.ndarray, t0: float) -> np.ndarray:
    """A time line with a small blurred bump at each compression."""
    n = int((times[-1] - t0) / GRID_S) + int(1.0 / GRID_S)
    idx = np.round((times - t0) / GRID_S).astype(int)
    train = np.bincount(idx, minlength=n).astype(float)
    return gaussian_filter1d(train, COARSE_SIGMA_S / GRID_S)


def _evaluate(g: np.ndarray, m: np.ndarray, tau: float, fit_drift: bool):
    """Fine-tune one candidate offset and score it."""
    a, b = 1.0, tau
    for _ in range(3):
        ei, ti = pair_events(g, a * m + b, PAIR_TOLERANCE_S)
        if ei.size < MIN_PAIRS:
            return None
        span = m[ti[-1]] - m[ti[0]]
        if fit_drift and ei.size >= 20 and span >= 30.0:
            a, b = _robust_line(m[ti], g[ei])
        else:
            b += float(np.median(g[ei] - (a * m[ti] + b)))

    ei, ti = pair_events(g, a * m + b, PAIR_TOLERANCE_S)
    if ei.size < MIN_PAIRS:
        return None
    resid = g[ei] - (a * m[ti] + b)
    resid_sd = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    score = float(np.sum(np.exp(-0.5 * (resid / FINE_SIGMA_S) ** 2)))
    return {"a": a, "b": b, "n": int(ei.size), "resid_sd": resid_sd, "score": score}


def _robust_line(x: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
    """Straight-line fit that ignores a few stray points."""
    x0 = float(np.mean(x))  # centre the times so large clock values keep precision
    xc = x - x0
    a, c = np.polyfit(xc, yv, 1)
    resid = yv - (a * xc + c)
    sd = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    keep = np.abs(resid - np.median(resid)) <= 3.0 * max(sd, 1e-4)
    if 10 <= np.count_nonzero(keep) < x.size:
        a, c = np.polyfit(xc[keep], yv[keep], 1)
    return float(a), float(c - a * x0)
