"""The bench test: score a whole folder of sessions and give the verdicts.

Answers the plan's bench-test question (do the glasses measure what the
manikin measures?) across many sessions and people, and the open
question behind depth: does one calibration hold for everyone?

What it needs from each session's session.json:
    person   a code such as "P1", so sessions can be grouped by person
    kind     "calibration" for a shallow-to-deep sweep, anything else is a test

Two kinds of result come out:
    measurement   how close the glasses' numbers are to the manikin's
    live coach    each test session is replayed through the live detector
                  and coach, and what the coach showed is checked against
                  the manikin. This is where the "never say good when it
                  is not" guardrail is judged.

Rate, pauses and depth are scored on the test sessions. Calibration
sessions are only used to fit depth calibrations, two ways:
    one for everyone   fitted on the OTHER people's calibration sessions,
                       so each person is new to the calibration that scores them
    per person         fitted on that person's own calibration sessions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .depth import Calibration, fit_calibration
from .live_eval import LiveResult, evaluate_live
from .scorecard import SessionScore, pair_session, score_session
from .scoring import (
    DEPTH_BAND_PASS_FRACTION,
    GUARDRAIL_MAX,
    RATE_PASS_FRACTION,
    DepthScore,
    score_depth,
)
from .session import POSE_FILE, SessionError, load_session

SESSIONS_PLANNED = 20       # the technical plan's bench test size
ONE_FOR_EVERYONE = "one calibration for everyone"
PER_PERSON = "calibrated per person"
GIVEN = "given calibration"


@dataclass
class BenchSession:
    session_id: str
    person: str
    kind: str
    is_calibration: bool
    score: SessionScore
    usable: bool                  # has a manikin file and a trusted alignment
    note: str = ""                # why it is not usable
    travel_m: np.ndarray | None = None      # paired head travel (usable sessions with depth)
    true_depth_m: np.ndarray | None = None  # paired manikin depth
    depth: dict[str, DepthScore] = field(default_factory=dict)   # by calibration mode
    depth_est_m: dict[str, np.ndarray] = field(default_factory=dict)   # estimates behind each score
    live: dict[str, LiveResult] = field(default_factory=dict)    # live-coach replay, by mode
    replay: SimpleNamespace | None = None   # what the replay needs: t, y, true_times, true_depths_m


@dataclass
class PersonSummary:
    person: str
    n_test_sessions: int
    travel_per_depth: float       # head travel per unit of chest depth (NaN if no depth)
    scatter: float                # compression-to-compression variation in that, as a fraction
    rate_segments: int
    rate_within: int
    depth: dict[str, DepthScore] = field(default_factory=dict)


@dataclass
class CoachCheck:
    """What the live coach showed across the test sessions, against the manikin."""

    n_sessions: int
    false_good: float          # average share of the time it said "good" wrongly
    false_good_worst: float    # the worst single session
    wrong_way: float           # average share of the time it pointed the wrong way
    needless: float            # average share of the time it corrected someone in range

    @property
    def passed(self) -> bool:
        return self.false_good < GUARDRAIL_MAX


@dataclass
class BenchReport:
    sessions: list[BenchSession]
    errors: list[tuple[str, str]]             # (folder, message) for folders that could not be read
    people: list[PersonSummary]
    # Pooled over the usable test sessions:
    rate_segments: int = 0
    rate_within: int = 0
    rate_false_in_range: int = 0
    depth: dict[str, DepthScore] = field(default_factory=dict)
    depth_note: str = ""                      # why depth was not scored, if it was not
    coach_rate: CoachCheck | None = None
    coach_depth: dict[str, CoachCheck] = field(default_factory=dict)
    people_differ_by: float = float("nan")    # variation in travel_per_depth between people
    scatter_within_person: float = float("nan")

    @property
    def test_sessions(self) -> list[BenchSession]:
        return [s for s in self.sessions if not s.is_calibration and s.score.alignment is not None]

    @property
    def usable_tests(self) -> list[BenchSession]:
        return [s for s in self.test_sessions if s.usable]

    @property
    def rate_passed(self) -> bool:
        return self.rate_segments > 0 and self.rate_within / self.rate_segments >= RATE_PASS_FRACTION

    @property
    def rate_guardrail_passed(self) -> bool:
        return self.rate_segments > 0 and self.rate_false_in_range / self.rate_segments < GUARDRAIL_MAX


def find_sessions(root) -> list[Path]:
    """Every folder under root (or root itself) that holds a head_pose.csv."""
    root = Path(root)
    if (root / POSE_FILE).exists():
        return [root]
    return sorted(p.parent for p in root.rglob(POSE_FILE))


def run_bench(folders, calibration: Calibration | None = None) -> BenchReport:
    sessions: list[BenchSession] = []
    errors: list[tuple[str, str]] = []

    for folder in folders:
        try:
            session = load_session(folder)
        except SessionError as e:
            errors.append((str(folder), str(e)))
            continue
        kind = str(session.meta.get("kind", "test")).lower()
        row = BenchSession(
            session_id=session.session_id,
            person=str(session.meta.get("person", "unknown")),
            kind=kind,
            is_calibration=kind == "calibration",
            score=score_session(session),
            usable=False,
        )
        if not session.has_answer_key:
            row.note = "no manikin file"
        else:
            p = pair_session(session)
            if not p.alignment.ok:
                row.note = "alignment not trusted: " + p.alignment.reason
            else:
                row.usable = True
                row.replay = SimpleNamespace(
                    t=session.t - session.t[0],
                    y=session.height,
                    true_times=p.true_times,
                    true_depths_m=p.true_depth_m,
                )
                if p.true_depth_m is not None:
                    travel = p.travel_m[p.est_idx]
                    truth = p.true_depth_m[p.true_idx]
                    ok = ~np.isnan(travel)
                    row.travel_m, row.true_depth_m = travel[ok], truth[ok]
        sessions.append(row)

    report = BenchReport(sessions=sessions, errors=errors, people=[])
    _pool_rate(report)
    calibrations = _score_depth(report, calibration)
    _replay_coach(report, calibrations)
    _summarise_people(report)
    for s in sessions:
        s.replay = None         # the raw signals are no longer needed
    return report


# ---- rate ---------------------------------------------------------------------


def _pool_rate(report: BenchReport) -> None:
    for s in report.usable_tests:
        r = s.score.rate
        if r is None:
            continue
        report.rate_segments += r.n_segments
        report.rate_within += r.n_within
        report.rate_false_in_range += r.n_false_reassurance


# ---- depth --------------------------------------------------------------------


def _has_depth(s: BenchSession) -> bool:
    return s.usable and s.travel_m is not None and s.travel_m.size >= 5


def _fit(sessions: list[BenchSession]) -> Calibration | None:
    sessions = [s for s in sessions if _has_depth(s)]
    if not sessions:
        return None
    try:
        return fit_calibration(
            np.concatenate([s.travel_m for s in sessions]),
            np.concatenate([s.true_depth_m for s in sessions]),
        )
    except ValueError:
        return None


def _score_depth(report: BenchReport, given: Calibration | None) -> dict:
    """Scores depth; returns the calibration used for each person under each mode."""
    tests = [s for s in report.usable_tests if _has_depth(s)]
    cal_sessions = [s for s in report.sessions if s.is_calibration and _has_depth(s)]
    if not tests:
        report.depth_note = "no usable test session has manikin depth"
        return {}

    plans: dict[str, dict[str, Calibration | None]] = {}   # mode -> person -> calibration
    people = sorted({s.person for s in tests})
    if given is not None:
        plans[GIVEN] = {p: given for p in people}
    elif not cal_sessions:
        report.depth_note = (
            'no calibration sessions found (session.json "kind": "calibration"), '
            "and no calibration file given"
        )
        return {}
    else:
        cal_people = {s.person for s in cal_sessions}
        if len(cal_people) >= 2:
            plans[ONE_FOR_EVERYONE] = {
                p: _fit([s for s in cal_sessions if s.person != p]) for p in people
            }
        else:
            report.depth_note = (
                "calibration sessions from only one person, so one calibration "
                "for everyone could not be tested on a new person"
            )
        plans[PER_PERSON] = {p: _fit([s for s in cal_sessions if s.person == p]) for p in people}

    for mode, by_person in plans.items():
        est_all, true_all = [], []
        for s in tests:
            cal = by_person.get(s.person)
            if cal is None:
                continue
            est = cal.depth(s.travel_m)
            s.depth[mode] = score_depth(est, s.true_depth_m)
            s.depth_est_m[mode] = est
            est_all.append(est)
            true_all.append(s.true_depth_m)
        if est_all:
            report.depth[mode] = score_depth(np.concatenate(est_all), np.concatenate(true_all))
    return plans


# ---- live coach replay ------------------------------------------------------------


def _replay_coach(report: BenchReport, calibrations: dict) -> None:
    tests = [s for s in report.usable_tests if s.replay is not None]
    if not tests:
        return

    # Rate cues do not depend on a depth calibration.
    for s in tests:
        s.live["rate"] = evaluate_live(s.replay, None)
    report.coach_rate = _coach_check(
        [s.live["rate"] for s in tests], "rate_false_good", "rate_wrong_way", "rate_needless"
    )

    for mode, by_person in calibrations.items():
        results = []
        for s in tests:
            cal = by_person.get(s.person)
            if cal is None or s.replay.true_depths_m is None:
                continue
            s.live[mode] = evaluate_live(s.replay, cal)
            results.append(s.live[mode])
        if results:
            report.coach_depth[mode] = _coach_check(
                results, "depth_false_good", "depth_wrong_way", "depth_needless"
            )


def _coach_check(results: list[LiveResult], false_good: str, wrong_way: str, needless: str):
    fg = np.array([getattr(r, false_good) for r in results], dtype=float)
    fg = fg[~np.isnan(fg)]
    if fg.size == 0:
        return None
    return CoachCheck(
        n_sessions=int(fg.size),
        false_good=float(fg.mean()),
        false_good_worst=float(fg.max()),
        wrong_way=float(np.nanmean([getattr(r, wrong_way) for r in results])),
        needless=float(np.nanmean([getattr(r, needless) for r in results])),
    )


# ---- people -------------------------------------------------------------------


def _summarise_people(report: BenchReport) -> None:
    by_person: dict[str, list[BenchSession]] = {}
    for s in report.sessions:
        if s.score.alignment is not None:
            by_person.setdefault(s.person, []).append(s)

    ratios, scatters = [], []
    for person in sorted(by_person):
        rows = by_person[person]
        tests = [s for s in rows if not s.is_calibration and s.usable]
        summary = PersonSummary(
            person=person,
            n_test_sessions=len(tests),
            travel_per_depth=float("nan"),
            scatter=float("nan"),
            rate_segments=sum(s.score.rate.n_segments for s in tests if s.score.rate),
            rate_within=sum(s.score.rate.n_within for s in tests if s.score.rate),
        )

        with_depth = [s for s in rows if _has_depth(s)]
        if with_depth:
            ratio = np.concatenate([s.travel_m / s.true_depth_m for s in with_depth])
            centre = float(np.median(ratio))
            summary.travel_per_depth = centre
            summary.scatter = float(1.4826 * np.median(np.abs(ratio - centre)) / centre)
            ratios.append(centre)
            scatters.append(summary.scatter)

        for mode in report.depth:
            scored = [s for s in tests if mode in s.depth_est_m]
            if scored:
                summary.depth[mode] = score_depth(
                    np.concatenate([s.depth_est_m[mode] for s in scored]),
                    np.concatenate([s.true_depth_m for s in scored]),
                )
        report.people.append(summary)

    if len(ratios) >= 2:
        report.people_differ_by = float(np.std(ratios, ddof=1) / np.mean(ratios))
    if scatters:
        report.scatter_within_person = float(np.median(scatters))
