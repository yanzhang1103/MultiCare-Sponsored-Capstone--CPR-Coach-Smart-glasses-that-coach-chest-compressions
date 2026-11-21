"""Score one recorded session from start to finish.

Steps: line up the two clocks, find the compressions, then score
rate, pauses and depth against the manikin. A session with no manikin
file (pillow practice) gets measurements only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import Alignment, align_events
from .depth import Calibration, fit_calibration, head_travel
from .rate import (
    PAUSE_THRESHOLD_S,
    detect_compressions,
    find_pauses,
    rate_in_window,
    rate_series,
    run_mask,
)
from .scoring import (
    DEPTH_BAND_PASS_FRACTION,
    GOOD_RATE_CPM,
    GUARDRAIL_MAX,
    DepthScore,
    EventScore,
    PauseScore,
    RateScore,
    depth_band,
    match_events,
    pair_events,
    score_depth,
    score_pauses,
    score_rate,
)
from .session import Session


@dataclass
class Measurements:
    """What the glasses measured, with no answer key involved."""

    n_compressions: int
    rate_cpm: float                    # typical rate over the session
    rate_in_range_fraction: float      # share of compression time with rate in the target range
    n_pauses: int
    pause_time_s: float
    depth_mean_m: float | None = None           # needs a calibration
    depth_in_range_fraction: float | None = None


@dataclass
class TravelVsDepth:
    """How well head travel tracks manikin depth within this one session.

    Fitted on the session itself, so it flatters: it is the first check
    that the idea works at all, not a score.
    """

    n: int
    correlation: float
    gain: float
    offset_m: float
    scatter_mm: float          # typical depth error left after the best fit
    depth_range_m: tuple[float, float]


@dataclass
class SessionScore:
    session_id: str
    duration_s: float
    meta: dict
    measurements: Measurements
    warnings: list[str] = field(default_factory=list)
    alignment: Alignment | None = None
    events: EventScore | None = None
    rate: RateScore | None = None
    pauses: PauseScore | None = None
    travel_vs_depth: TravelVsDepth | None = None
    depth: DepthScore | None = None


@dataclass
class PairedSession:
    """Detected compressions paired with the manikin's, on the glasses clock."""

    alignment: Alignment
    est_times: np.ndarray        # detected compressions (start cue left out)
    true_times: np.ndarray       # manikin compressions on the glasses clock (start cue left out)
    true_depth_m: np.ndarray | None
    est_idx: np.ndarray          # est_times[est_idx[k]] pairs with true_times[true_idx[k]]
    true_idx: np.ndarray
    travel_m: np.ndarray         # head travel for every detected compression


def pair_session(session: Session) -> PairedSession:
    """Align the clocks and pair the glasses' compressions with the manikin's."""
    if not session.has_answer_key:
        raise ValueError(f"{session.session_id} has no manikin file to pair with")
    t = session.t - session.t[0]
    y = session.height

    # Runs of three are allowed here so the start cue helps the alignment.
    alignment = align_events(detect_compressions(t, y, min_run=3), session.manikin_t)

    est = detect_compressions(t, y)
    travel = head_travel(t, y, est)

    if alignment.n_paired == 0:
        empty = np.array([], dtype=int)
        return PairedSession(alignment, est, np.array([]), None, empty, empty, travel)

    true_all = alignment.to_glasses_time(session.manikin_t)
    # The detector ignores runs shorter than four (the start cue), so the
    # answer key leaves them out too.
    keep = run_mask(true_all)
    true = true_all[keep]
    depth = session.manikin_depth_m[keep] if session.manikin_depth_m is not None else None
    est_idx, true_idx = pair_events(est, true)
    return PairedSession(alignment, est, true, depth, est_idx, true_idx, travel)


def score_session(session: Session, calibration: Calibration | None = None) -> SessionScore:
    t = session.t - session.t[0]
    y = session.height
    duration = float(t[-1])
    warnings = list(session.warnings)

    if not session.has_answer_key:
        est = detect_compressions(t, y)
        travel = head_travel(t, y, est)
        return SessionScore(
            session_id=session.session_id,
            duration_s=duration,
            meta=session.meta,
            measurements=_measure(est, travel, duration, calibration),
            warnings=warnings,
        )

    p = pair_session(session)
    score = SessionScore(
        session_id=session.session_id,
        duration_s=duration,
        meta=session.meta,
        measurements=_measure(p.est_times, p.travel_m, duration, calibration),
        warnings=warnings,
        alignment=p.alignment,
    )
    if p.alignment.n_paired == 0:
        return score

    score.events = match_events(p.est_times, p.true_times)
    score.rate = score_rate(p.est_times, p.true_times, duration)
    score.pauses = score_pauses(p.est_times, p.true_times)

    # Depth pairs one compression with one compression, so it is only
    # scored when the alignment is trusted.
    if p.alignment.ok and p.true_depth_m is not None and p.est_idx.size >= 5:
        travel = p.travel_m[p.est_idx]
        truth = p.true_depth_m[p.true_idx]
        ok = ~np.isnan(travel)
        travel, truth = travel[ok], truth[ok]
        score.travel_vs_depth = _travel_vs_depth(travel, truth)
        if calibration is not None:
            score.depth = score_depth(calibration.depth(travel), truth)
    return score


def _measure(est, travel, duration, calibration) -> Measurements:
    if est.size >= 2:
        t_eval = np.arange(5.0, max(duration, 5.5), 0.5)
        rolling = rate_series(est, t_eval)
        valid = rolling[~np.isnan(rolling)]
        in_range = (
            float(np.mean((valid >= GOOD_RATE_CPM[0]) & (valid <= GOOD_RATE_CPM[1])))
            if valid.size
            else float("nan")
        )
    else:
        in_range = float("nan")
    pauses = find_pauses(est, PAUSE_THRESHOLD_S)
    m = Measurements(
        n_compressions=int(est.size),
        rate_cpm=rate_in_window(est, -np.inf, np.inf),
        rate_in_range_fraction=in_range,
        n_pauses=len(pauses),
        pause_time_s=float(sum(b - a for a, b in pauses)),
    )
    if calibration is not None and est.size:
        depth = calibration.depth(travel[~np.isnan(travel)])
        if depth.size:
            m.depth_mean_m = float(np.mean(depth))
            m.depth_in_range_fraction = float(np.mean(depth_band(depth) == 0))
    return m


def _travel_vs_depth(travel: np.ndarray, truth: np.ndarray) -> TravelVsDepth | None:
    if travel.size < 5 or np.std(travel) == 0 or np.std(truth) == 0:
        return None
    cal = fit_calibration(travel, truth)
    resid = truth - cal.depth(travel)
    return TravelVsDepth(
        n=int(travel.size),
        correlation=float(np.corrcoef(travel, truth)[0, 1]),
        gain=cal.gain,
        offset_m=cal.offset_m,
        scatter_mm=float(np.mean(np.abs(resid)) * 1000.0),
        depth_range_m=(float(truth.min()), float(truth.max())),
    )


# ---- printing ----------------------------------------------------------------


def format_scorecard(s: SessionScore) -> str:
    lines: list[str] = []
    meta = s.meta
    about = ", ".join(
        str(meta[k]) for k in ("person", "device", "surface", "kind") if meta.get(k)
    )
    lines.append(f"Session {s.session_id}" + (f"  ({about})" if about else ""))
    lines.append(f"  length {s.duration_s:.0f} s")
    for w in s.warnings:
        lines.append(f"  WARNING  {w}")
    lines.append("")

    def row(label: str, verdict: str, text: str) -> None:
        lines.append(f"  {label:<25}{verdict:<9}{text}")

    m = s.measurements
    lines.append("What the glasses measured")
    row("compressions", "", f"{m.n_compressions}")
    row("rate", "", f"{_num(m.rate_cpm, '.0f')} per minute; in the 100 to 120 range "
                    f"{_pct(m.rate_in_range_fraction)} of the time")
    row("pauses", "", f"{m.n_pauses}, {m.pause_time_s:.0f} s in total")
    if m.depth_mean_m is not None:
        row("depth", "", f"{m.depth_mean_m * 100:.1f} cm on average; in the 5 to 6 cm range "
                         f"{_pct(m.depth_in_range_fraction)} of compressions")
    else:
        row("depth", "", "not estimated (no calibration given)")

    if s.alignment is None:
        lines.append("")
        lines.append("No manikin file, so nothing to score against.")
        return "\n".join(lines)

    a = s.alignment
    lines.append("")
    lines.append("Against the manikin")
    if a.n_paired == 0:
        row("alignment", "FAILED", a.reason)
        row("", "", "Nothing was scored.")
        return "\n".join(lines)

    row("alignment", "OK" if a.ok else "FLAGGED",
        f"clocks {a.offset_s:+.3f} s apart, drift {a.drift_ppm:+.0f} ppm, "
        f"{a.paired_fraction:.0%} of manikin compressions matched")
    if not a.ok:
        row("", "", a.reason)
        row("", "", "Rate and pauses are shown but unconfirmed; depth is not scored.")

    e, r, p = s.events, s.rate, s.pauses
    row("compressions", "", f"{e.n_true} on the manikin, {e.n_detected} found; "
                            f"{e.n_missed} missed, {e.n_extra} extra (start cue left out)")
    if r.n_segments:
        row("rate", _verdict(r.passed),
            f"{r.n_within} of {r.n_segments} 30-second segments within 5 per minute "
            f"(pass mark 90%); average error {r.mean_abs_error_cpm:.1f}")
        row('rate, false "in range"', _verdict(r.guardrail_passed),
            f"{r.n_false_reassurance} of {r.n_segments} segments (limit {GUARDRAIL_MAX:.0%})")
    else:
        row("rate", "", "not scored: no segment had enough compressions")
    row("pauses", "", f"{p.n_found} of {p.n_true} found; {p.n_false} false")

    tv = s.travel_vs_depth
    d = s.depth
    if d is not None:
        row("depth", _verdict(d.passed),
            f"right band in {d.band_accuracy:.0%} of compressions "
            f"(proposed pass mark {DEPTH_BAND_PASS_FRACTION:.0%}); "
            f"average error {d.mean_abs_error_mm:.1f} mm, bias {d.bias_mm:+.1f} mm")
        row('depth, false "good"', _verdict(d.guardrail_passed),
            f"{d.false_reassurance_fraction:.1%} of compressions (limit {GUARDRAIL_MAX:.0%})")
    elif a.ok and tv is not None:
        row("depth", "", "not scored (no calibration given)")
    elif a.ok:
        row("depth", "", "not scored (the manikin file has no usable depth)")
    if tv is not None:
        row("head travel vs depth", "",
            f"correlation {tv.correlation:.2f}; best-fit error {tv.scatter_mm:.1f} mm over depths "
            f"{tv.depth_range_m[0] * 100:.1f} to {tv.depth_range_m[1] * 100:.1f} cm")
        row("", "", "(fitted on this session itself, so it flatters)")
    return "\n".join(lines)


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _pct(x: float) -> str:
    return "-" if x is None or np.isnan(x) else f"{x:.0%}"


def _num(x: float, fmt: str) -> str:
    return "-" if x is None or np.isnan(x) else format(x, fmt)
