import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_report import format_report, write_csv  # noqa: E402
from cpr_engine import Calibration, save_session, simulate_manikin_clock, simulate_session  # noqa: E402
from cpr_engine.bench import (  # noqa: E402
    GIVEN,
    ONE_FOR_EVERYONE,
    PER_PERSON,
    find_sessions,
    run_bench,
)

GAINS = {"P1": 0.75, "P2": 0.90, "P3": 1.05}


def _write(root, session_id, person, kind, sim, with_manikin=True, seed=0):
    n = sim.t.size
    pos = np.column_stack((np.zeros(n), 0.75 + sim.y, np.zeros(n)))
    manikin_t = manikin_depth = None
    if with_manikin:
        manikin_t = simulate_manikin_clock(sim.true_times, offset_s=5.0 + seed, seed=seed)
        manikin_depth = sim.true_depths_m
    save_session(root / session_id, {"session_id": session_id, "person": person, "kind": kind},
                 sim.t, pos, None, manikin_t, manikin_depth)


def _make_bench(root, people=GAINS, with_sweeps=True):
    seed = 0
    for person, gain in people.items():
        args = dict(head_gain=gain, head_coupling_jitter=0.03, start_cue=3)
        if with_sweeps:
            seed += 1
            sweep = simulate_session(duration_s=60, depth_m=0.035, depth_drift_m=0.035,
                                     depth_jitter=0.05, seed=seed, **args)
            _write(root, f"{person}_sweep", person, "calibration", sweep, seed=seed)
        for k, (rate, depth) in enumerate(((110, 0.055), (90, 0.043))):
            seed += 1
            sim = simulate_session(duration_s=60, rate_cpm=rate, depth_m=depth, seed=seed, **args)
            _write(root, f"{person}_test{k + 1}", person, "test", sim, seed=seed)
    return root


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = _make_bench(tmp_path_factory.mktemp("bench"))
    return root, run_bench(find_sessions(root))


def test_counts_and_rate(bench):
    _, r = bench
    assert len(r.sessions) == 9 and not r.errors
    assert len(r.test_sessions) == 6 and len(r.usable_tests) == 6
    assert r.rate_segments == 12 and r.rate_passed


def test_depth_is_scored_both_ways(bench):
    _, r = bench
    assert set(r.depth) == {ONE_FOR_EVERYONE, PER_PERSON}
    one, per = r.depth[ONE_FOR_EVERYONE], r.depth[PER_PERSON]
    assert per.mean_abs_error_mm < 3.0
    assert one.mean_abs_error_mm > per.mean_abs_error_mm + 1.0    # these people differ a lot
    assert abs(per.bias_mm) < 1.0


def test_one_for_everyone_never_sees_the_person_it_scores(bench):
    _, r = bench
    by = {p.person: p for p in r.people}
    # P1's head moves least, so a calibration from the others reads P1 too shallow.
    assert by["P1"].depth[ONE_FOR_EVERYONE].bias_mm < -2.0
    assert by["P3"].depth[ONE_FOR_EVERYONE].bias_mm > 2.0


def test_person_numbers(bench):
    _, r = bench
    by = {p.person: p for p in r.people}
    for person, gain in GAINS.items():
        assert by[person].travel_per_depth == pytest.approx(gain, abs=0.03)
        assert by[person].n_test_sessions == 2
    assert r.people_differ_by == pytest.approx(np.std(list(GAINS.values()), ddof=1) / 0.9, abs=0.03)
    assert 0.02 < r.scatter_within_person < 0.08


def test_live_coach_is_replayed(bench):
    _, r = bench
    assert r.coach_rate is not None and r.coach_rate.n_sessions == 6
    assert r.coach_rate.false_good < 0.01 and r.coach_rate.wrong_way == 0.0
    assert set(r.coach_depth) == {ONE_FOR_EVERYONE, PER_PERSON}
    assert r.coach_depth[PER_PERSON].passed
    assert r.coach_depth[PER_PERSON].wrong_way < 0.01


def test_report_text_and_csv(bench, tmp_path):
    root, r = bench
    text = format_report(r, str(root))
    assert "By person" in text and "By session" in text and "Live coach" in text
    assert "PASS" in text
    write_csv(r, tmp_path / "out.csv")
    rows = (tmp_path / "out.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + 9
    assert rows[0].startswith("session,person,kind")


def test_given_calibration(bench):
    root, _ = bench
    r = run_bench(find_sessions(root), Calibration(gain=1.11, offset_m=0.0))
    assert set(r.depth) == {GIVEN}
    assert set(r.coach_depth) == {GIVEN}


def test_no_calibration_sessions(tmp_path):
    root = _make_bench(tmp_path, people={"P1": 0.9}, with_sweeps=False)
    r = run_bench(find_sessions(root))
    assert r.rate_passed
    assert not r.depth and "no calibration sessions" in r.depth_note
    assert "not scored" in format_report(r, str(root))


def test_one_person_only_gets_per_person(tmp_path):
    root = _make_bench(tmp_path, people={"P1": 0.9})
    r = run_bench(find_sessions(root))
    assert set(r.depth) == {PER_PERSON}
    assert "only one person" in r.depth_note


def test_bad_and_pillow_sessions_do_not_stop_the_report(tmp_path):
    root = _make_bench(tmp_path, people={"P1": 0.9, "P2": 1.0})
    (root / "broken").mkdir()
    (root / "broken" / "head_pose.csv").write_text("t_s,y_m\n")
    pillow = simulate_session(duration_s=40, seed=99)
    _write(root, "P1_pillow", "P1", "practice", pillow, with_manikin=False)
    steady = simulate_session(duration_s=40, rate_jitter=0.005, seed=98)   # cannot be aligned
    _write(root, "P2_steady", "P2", "test", steady, seed=98)

    r = run_bench(find_sessions(root))
    assert len(r.errors) == 1 and "broken" in r.errors[0][0]
    by = {s.session_id: s for s in r.sessions}
    assert not by["P1_pillow"].usable and by["P1_pillow"].note == "no manikin file"
    assert not by["P2_steady"].usable and "alignment" in by["P2_steady"].note
    assert len(r.usable_tests) == 4
    text = format_report(r, str(root))
    assert "COULD NOT READ" in text and "NOT USED" in text


def test_find_sessions_accepts_a_single_session(tmp_path):
    root = _make_bench(tmp_path, people={"P1": 0.9})
    assert find_sessions(root / "P1_test1") == [root / "P1_test1"]
    assert len(find_sessions(root)) == 3
