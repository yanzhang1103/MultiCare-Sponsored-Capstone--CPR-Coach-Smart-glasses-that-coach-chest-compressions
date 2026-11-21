"""Score the glasses' measurements against the answer key.

The answer key is the time of each true compression: from the
simulator now, from the Brayden Pro manikin later.

Pass marks come from the technical plan:
- Rate within +/-5 per minute in 90% of 30-second segments.
- Depth (proposed in the plan, not yet agreed): the correct depth band
  in 85% of compressions.
- Guardrail: the glasses say "you're doing it right" when the answer
  key says otherwise in fewer than 5% of cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rate import PAUSE_THRESHOLD_S, find_pauses, rate_in_window

SEGMENT_S = 30.0
RATE_TOLERANCE_CPM = 5.0
RATE_PASS_FRACTION = 0.90
GOOD_RATE_CPM = (100.0, 120.0)   # guideline range for compression rate
GUARDRAIL_MAX = 0.05
MATCH_TOLERANCE_S = 0.15
GOOD_DEPTH_M = (0.05, 0.06)      # guideline range for compression depth, adults
DEPTH_BAND_PASS_FRACTION = 0.85


@dataclass
class RateScore:
    n_segments: int               # segments with enough true compressions to score
    n_within: int                 # segments where the estimate is within tolerance
    fraction_within: float
    passed: bool                  # meets the 90% pass mark
    mean_abs_error_cpm: float
    n_false_reassurance: int      # glasses say in range, answer key says not
    false_reassurance_fraction: float
    guardrail_passed: bool
    segments: list[dict] = field(default_factory=list)


def score_rate(
    est_times,
    true_times,
    duration_s: float,
    segment_s: float = SEGMENT_S,
    tolerance_cpm: float = RATE_TOLERANCE_CPM,
    pass_fraction: float = RATE_PASS_FRACTION,
    good_range_cpm: tuple[float, float] = GOOD_RATE_CPM,
    guardrail_max: float = GUARDRAIL_MAX,
    pause_threshold_s: float = PAUSE_THRESHOLD_S,
) -> RateScore:
    """Compare estimated and true rate, segment by segment.

    A segment is scored only if the answer key has enough compressions
    in it to give a rate. If the glasses give no rate for a scored
    segment, that segment counts as a miss.
    A final segment shorter than half a segment is left out.
    """
    segments = []
    start = 0.0
    while start < duration_s:
        end = min(start + segment_s, duration_s)
        if (end - start) < segment_s / 2.0:
            break
        true_rate = rate_in_window(true_times, start, end, pause_threshold_s)
        if not np.isnan(true_rate):
            est_rate = rate_in_window(est_times, start, end, pause_threshold_s)
            error = abs(est_rate - true_rate) if not np.isnan(est_rate) else float("nan")
            within = (not np.isnan(error)) and error <= tolerance_cpm
            est_good = (not np.isnan(est_rate)) and _in_range(est_rate, good_range_cpm)
            true_good = _in_range(true_rate, good_range_cpm)
            segments.append(
                {
                    "start_s": start,
                    "end_s": end,
                    "true_cpm": true_rate,
                    "est_cpm": est_rate,
                    "error_cpm": error,
                    "within": within,
                    "false_reassurance": est_good and not true_good,
                }
            )
        start += segment_s

    n = len(segments)
    n_within = sum(s["within"] for s in segments)
    n_false = sum(s["false_reassurance"] for s in segments)
    errors = [s["error_cpm"] for s in segments if not np.isnan(s["error_cpm"])]
    frac = n_within / n if n else float("nan")
    false_frac = n_false / n if n else float("nan")
    return RateScore(
        n_segments=n,
        n_within=n_within,
        fraction_within=frac,
        passed=bool(n and frac >= pass_fraction),
        mean_abs_error_cpm=float(np.mean(errors)) if errors else float("nan"),
        n_false_reassurance=n_false,
        false_reassurance_fraction=false_frac,
        guardrail_passed=bool(n and false_frac < guardrail_max),
        segments=segments,
    )


def _in_range(value: float, rng: tuple[float, float]) -> bool:
    return rng[0] <= value <= rng[1]


@dataclass
class EventScore:
    n_true: int
    n_detected: int
    n_matched: int
    n_missed: int     # true compressions the glasses did not count
    n_extra: int      # compressions the glasses counted that did not happen


def pair_events(
    est_times, true_times, tolerance_s: float = MATCH_TOLERANCE_S
) -> tuple[np.ndarray, np.ndarray]:
    """Pair detected and true compressions that are within tolerance_s.

    Both inputs must be in time order. Returns two index arrays of the
    same length: est_times[est_idx[k]] is paired with true_times[true_idx[k]].
    """
    est = np.asarray(est_times, dtype=float)
    true = np.asarray(true_times, dtype=float)
    est_idx: list[int] = []
    true_idx: list[int] = []
    i = j = 0
    while i < est.size and j < true.size:
        d = est[i] - true[j]
        if abs(d) <= tolerance_s:
            est_idx.append(i)
            true_idx.append(j)
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return np.array(est_idx, dtype=int), np.array(true_idx, dtype=int)


def match_events(
    est_times, true_times, tolerance_s: float = MATCH_TOLERANCE_S
) -> EventScore:
    """Count detected compressions that pair with a true one within tolerance_s."""
    est = np.sort(np.asarray(est_times, dtype=float))
    true = np.sort(np.asarray(true_times, dtype=float))
    matched = pair_events(est, true, tolerance_s)[0].size
    return EventScore(
        n_true=int(true.size),
        n_detected=int(est.size),
        n_matched=int(matched),
        n_missed=int(true.size - matched),
        n_extra=int(est.size - matched),
    )


@dataclass
class PauseScore:
    n_true: int
    n_found: int      # true pauses the glasses also flagged
    n_missed: int
    n_false: int      # pauses the glasses flagged that did not happen


def score_pauses(
    est_times, true_times, pause_threshold_s: float = PAUSE_THRESHOLD_S
) -> PauseScore:
    """A true pause counts as found if a detected pause covers at least half of it."""
    true_p = find_pauses(true_times, pause_threshold_s)
    est_p = find_pauses(est_times, pause_threshold_s)

    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    found = sum(
        any(overlap(tp, ep) >= 0.5 * (tp[1] - tp[0]) for ep in est_p) for tp in true_p
    )
    false = sum(all(overlap(ep, tp) == 0.0 for tp in true_p) for ep in est_p)
    return PauseScore(
        n_true=len(true_p), n_found=found, n_missed=len(true_p) - found, n_false=false
    )


@dataclass
class DepthScore:
    n: int                            # compressions scored
    mean_abs_error_mm: float
    bias_mm: float                    # positive = glasses read too deep
    p90_abs_error_mm: float           # 90% of compressions are closer than this
    band_accuracy: float              # same band (shallow / good / deep) as the answer key
    passed: bool                      # meets the proposed 85% band pass mark
    false_reassurance_fraction: float  # glasses say good, answer key says not
    guardrail_passed: bool


def depth_band(depth_m, good_range_m: tuple[float, float] = GOOD_DEPTH_M) -> np.ndarray:
    """-1 = too shallow, 0 = good, 1 = too deep."""
    d = np.asarray(depth_m, dtype=float)
    return np.where(d < good_range_m[0], -1, np.where(d > good_range_m[1], 1, 0))


def score_depth(
    est_depth_m,
    true_depth_m,
    good_range_m: tuple[float, float] = GOOD_DEPTH_M,
    pass_fraction: float = DEPTH_BAND_PASS_FRACTION,
    guardrail_max: float = GUARDRAIL_MAX,
    say_good_margin_m: float = 0.0,
) -> DepthScore:
    """Compare estimated and true depth, compression by compression.

    The two inputs are already paired (see pair_events).
    say_good_margin_m: the glasses say "good" only when the estimate is
    this far inside the good range. Zero means the plain range.
    """
    est = np.asarray(est_depth_m, dtype=float)
    true = np.asarray(true_depth_m, dtype=float)
    if est.size != true.size:
        raise ValueError("estimated and true depth must be paired, same length")
    n = int(est.size)
    if n == 0:
        nan = float("nan")
        return DepthScore(0, nan, nan, nan, nan, False, nan, False)

    err = est - true
    band_acc = float(np.mean(depth_band(est, good_range_m) == depth_band(true, good_range_m)))
    says_good = (est >= good_range_m[0] + say_good_margin_m) & (
        est <= good_range_m[1] - say_good_margin_m
    )
    truly_good = depth_band(true, good_range_m) == 0
    false_frac = float(np.mean(says_good & ~truly_good))
    return DepthScore(
        n=n,
        mean_abs_error_mm=float(np.mean(np.abs(err)) * 1000),
        bias_mm=float(np.mean(err) * 1000),
        p90_abs_error_mm=float(np.percentile(np.abs(err), 90) * 1000),
        band_accuracy=band_acc,
        passed=band_acc >= pass_fraction,
        false_reassurance_fraction=false_frac,
        guardrail_passed=false_frac < guardrail_max,
    )
