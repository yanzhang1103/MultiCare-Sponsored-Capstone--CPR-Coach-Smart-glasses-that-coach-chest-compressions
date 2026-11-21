import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine import (  # noqa: E402
    detect_compressions,
    find_pauses,
    match_events,
    rate_in_window,
    score_pauses,
    score_rate,
    simulate_session,
)
from scenarios import SCENARIOS  # noqa: E402


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_scenario_meets_pass_marks(name, seed):
    s = simulate_session(seed=seed, **SCENARIOS[name])
    est = detect_compressions(s.t, s.y)
    rs = score_rate(est, s.true_times, s.duration_s)
    ps = score_pauses(est, s.true_times)
    assert rs.n_segments >= 3
    assert rs.passed, f"{rs.n_within}/{rs.n_segments} segments within tolerance"
    assert rs.guardrail_passed
    assert ps.n_missed == 0
    assert ps.n_false == 0


def test_no_compressions_in_stillness():
    s = simulate_session(rate_cpm=110, pauses=((0.0, 200.0),), n_glances=6)
    assert s.true_times.size == 0
    assert detect_compressions(s.t, s.y).size == 0


def test_rate_in_window_exact():
    times = np.arange(0, 30, 0.5)  # 120 per minute
    assert rate_in_window(times, 0, 30) == pytest.approx(120.0)


def test_rate_ignores_pause_gap():
    times = np.concatenate((np.arange(0, 10, 0.5), np.arange(15, 25, 0.5)))
    assert rate_in_window(times, 0, 30) == pytest.approx(120.0)
    assert find_pauses(times) == [(9.5, 15.0)]


def test_rate_nan_when_too_few():
    assert np.isnan(rate_in_window([1.0, 1.5], 0, 30))


def test_match_events_counts():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    est = np.array([1.05, 2.4, 3.0, 4.1, 5.0])
    ev = match_events(est, true)
    assert (ev.n_matched, ev.n_missed, ev.n_extra) == (3, 1, 2)


def test_guardrail_flags_false_reassurance():
    true = np.arange(0, 60, 60 / 140)   # truly too fast
    est = np.arange(0, 60, 60 / 110)    # glasses think it is in range
    rs = score_rate(est, true, 60.0)
    assert not rs.passed
    assert rs.n_false_reassurance == 2
    assert not rs.guardrail_passed
