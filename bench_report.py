r"""Score a whole folder of sessions and print the bench-test report.

    python bench_report.py sessions\bench
    python bench_report.py sessions\bench --csv bench_sessions.csv
    python bench_report.py sessions\bench --calibration calibration.json

Each session.json needs "person" (a code such as "P1") and "kind"
("calibration" for a shallow-to-deep sweep; anything else is a test).
Without --calibration, depth is scored two ways from the calibration
sessions: one calibration for everyone, and one per person.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from cpr_engine import SessionError, load_calibration
from cpr_engine.bench import (
    SESSIONS_PLANNED,
    BenchReport,
    find_sessions,
    run_bench,
)
from cpr_engine.scoring import DEPTH_BAND_PASS_FRACTION, GUARDRAIL_MAX, RATE_PASS_FRACTION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="folder holding the session folders")
    parser.add_argument("--calibration", help="score depth with this calibration file instead")
    parser.add_argument("--csv", help="also write one row per session to this file")
    args = parser.parse_args()

    folders = find_sessions(args.root)
    if not folders:
        print(f"ERROR  no sessions found under {args.root}")
        return 1
    try:
        calibration = load_calibration(args.calibration) if args.calibration else None
    except SessionError as e:
        print(f"ERROR  {e}")
        return 1

    report = run_bench(folders, calibration)
    print(format_report(report, args.root))
    if args.csv:
        write_csv(report, args.csv)
        print(f"\nsaved {args.csv}")
    return 0


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _pct(x: float, digits: int = 0) -> str:
    return "-" if x is None or np.isnan(x) else f"{x:.{digits}%}"


def format_report(r: BenchReport, root: str) -> str:
    lines: list[str] = []
    tests = r.test_sessions
    usable = r.usable_tests
    n_cal = sum(s.is_calibration for s in r.sessions)
    n_no_key = sum(s.score.alignment is None for s in r.sessions)
    people = sorted({s.person for s in r.sessions})

    lines.append(f"Bench test report: {root}")
    lines.append(
        f"  {len(r.sessions)} sessions read from {len(people)} "
        f"{'person' if len(people) == 1 else 'people'}: {len(tests)} test, {n_cal} calibration"
        + (f", {n_no_key} with no manikin file" if n_no_key else "")
    )
    lines.append(
        f"  {len(usable)} of {len(tests)} test sessions usable "
        f"(the plan calls for {SESSIONS_PLANNED})"
    )
    for folder, message in r.errors:
        lines.append(f"  COULD NOT READ  {folder}: {message}")
    for s in r.sessions:
        if s.score.alignment is not None and not s.usable:
            lines.append(f"  NOT USED  {s.session_id}: {s.note}")

    def row(label: str, verdict: str, text: str) -> None:
        lines.append(f"  {label:<40}{verdict:<7}{text}")

    lines.append("")
    lines.append("Measurement: the glasses' numbers against the manikin's (usable test sessions pooled)")
    if r.rate_segments:
        row("rate", _verdict(r.rate_passed),
            f"{r.rate_within} of {r.rate_segments} 30-second segments within 5 per minute "
            f"({r.rate_within / r.rate_segments:.0%}; pass mark {RATE_PASS_FRACTION:.0%})")
    else:
        row("rate", "", "not scored: no usable test sessions")
    n_missed = sum(s.score.events.n_missed for s in usable)
    n_extra = sum(s.score.events.n_extra for s in usable)
    n_true = sum(s.score.events.n_true for s in usable)
    p_found = sum(s.score.pauses.n_found for s in usable)
    p_true = sum(s.score.pauses.n_true for s in usable)
    p_false = sum(s.score.pauses.n_false for s in usable)
    if usable:
        row("compressions", "", f"{n_true} on the manikin; {n_missed} missed, {n_extra} extra")
        row("pauses", "", f"{p_found} of {p_true} found; {p_false} false")
    for mode, d in r.depth.items():
        row(f"depth, {mode}", _verdict(d.passed),
            f"right band in {d.band_accuracy:.0%} of {d.n} compressions "
            f"(proposed pass mark {DEPTH_BAND_PASS_FRACTION:.0%}); "
            f"average error {d.mean_abs_error_mm:.1f} mm, bias {d.bias_mm:+.1f} mm")
    if r.depth_note:
        row("depth", "", ("not scored: " if not r.depth else "note: ") + r.depth_note)

    lines.append("")
    lines.append("Live coach: each test session replayed, and what the coach showed checked against the manikin")

    def coach_row(label: str, c) -> None:
        if c is None:
            return
        row(label, _verdict(c.passed),
            f'said "good" wrongly {c.false_good:.1%} of the time (worst session '
            f"{c.false_good_worst:.1%}; limit {GUARDRAIL_MAX:.0%}); pointed the wrong way "
            f"{c.wrong_way:.1%}; needless corrections {c.needless:.1%}")

    coach_row("rate cue", r.coach_rate)
    for mode, c in r.coach_depth.items():
        coach_row(f"depth cue, {mode}", c)
    if r.coach_rate is None:
        row("coach", "", "not checked: no usable test sessions")

    modes = list(r.depth)
    if r.people:
        lines.append("")
        lines.append("By person")
        head = f"  {'person':<10}{'tests':>6}{'rate within 5':>15}{'head travel':>13}{'scatter':>9}"
        for mode in modes:
            head += f"{_short(mode) + ' err mm':>20}{'bias':>7}"
        lines.append(head)
        for p in r.people:
            rate = f"{p.rate_within}/{p.rate_segments}" if p.rate_segments else "-"
            travel = "-" if np.isnan(p.travel_per_depth) else f"{p.travel_per_depth:.2f}"
            line = f"  {p.person:<10}{p.n_test_sessions:>6}{rate:>15}{travel:>13}{_pct(p.scatter, 1):>9}"
            for mode in modes:
                d = p.depth.get(mode)
                line += (
                    f"{d.mean_abs_error_mm:>20.1f}{d.bias_mm:>+7.1f}" if d else f"{'-':>20}{'-':>7}"
                )
            lines.append(line)
        lines.append("  head travel: cm the head moves per cm of chest depth")
        lines.append("  scatter:     how much that varies from one compression to the next")
        if not np.isnan(r.people_differ_by):
            lines.append("")
            lines.append(
                f"  People differ by {r.people_differ_by:.1%} in head travel per cm of depth. "
                "In simulation, one calibration"
            )
            lines.append("  for everyone held the 85% band mark up to about 6%.")
        if not np.isnan(r.scatter_within_person):
            lines.append(
                f"  Within one person, head travel scatters by {r.scatter_within_person:.1%} "
                "from compression to compression."
            )

    lines.append("")
    lines.append("By session")
    head = f"  {'session':<26}{'person':<8}{'kind':<13}{'missed':>7}{'extra':>6}{'rate segs':>11}"
    for mode in modes:
        head += f"{_short(mode) + ' err mm':>20}{'band':>6}"
    lines.append(head)
    for s in r.sessions:
        line = f"  {s.session_id[:25]:<26}{s.person[:7]:<8}{s.kind[:12]:<13}"
        if not s.usable:
            lines.append(line + f"  ({s.note})")
            continue
        e, rate = s.score.events, s.score.rate
        segs = f"{rate.n_within}/{rate.n_segments}" if rate and rate.n_segments else "-"
        line += f"{e.n_missed:>7}{e.n_extra:>6}{segs:>11}"
        for mode in modes:
            d = s.depth.get(mode)
            line += f"{d.mean_abs_error_mm:>20.1f}{d.band_accuracy:>6.0%}" if d else f"{'-':>20}{'-':>6}"
        lines.append(line)
    return "\n".join(lines)


def _short(mode: str) -> str:
    return {"one calibration for everyone": "one-for-all", "calibrated per person": "per-person",
            "given calibration": "given"}.get(mode, mode)


def write_csv(r: BenchReport, path: str) -> None:
    modes = list(r.depth)
    header = ["session", "person", "kind", "usable", "note", "length_s", "compressions_manikin",
              "missed", "extra", "rate_segments", "rate_segments_within_5", "rate_avg_error_cpm",
              "pauses_manikin", "pauses_found", "pauses_false", "coach_rate_false_good"]
    for mode in modes:
        m = _short(mode)
        header += [f"depth_{m}_avg_error_mm", f"depth_{m}_bias_mm", f"depth_{m}_band_right",
                   f"coach_depth_{m}_false_good"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in r.sessions:
            row = [s.session_id, s.person, s.kind, "yes" if s.usable else "no", s.note,
                   f"{s.score.duration_s:.0f}"]
            if s.usable:
                e, rate, p = s.score.events, s.score.rate, s.score.pauses
                live_rate = s.live.get("rate")
                row += [e.n_true, e.n_missed, e.n_extra, rate.n_segments, rate.n_within,
                        f"{rate.mean_abs_error_cpm:.2f}", p.n_true, p.n_found, p.n_false,
                        f"{live_rate.rate_false_good:.3f}" if live_rate else ""]
                for mode in modes:
                    d = s.depth.get(mode)
                    live = s.live.get(mode)
                    row += ([f"{d.mean_abs_error_mm:.2f}", f"{d.bias_mm:.2f}",
                             f"{d.band_accuracy:.3f}",
                             f"{live.depth_false_good:.3f}" if live else ""]
                            if d else ["", "", "", ""])
            w.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
