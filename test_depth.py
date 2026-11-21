import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine import (  # noqa: E402
    Calibration,
    depth_band,
    detect_compressions,
    fit_calibration,
    head_travel,
    pair_events,
    score_depth,
    simulate_session,
)
from depth_experiment import (  # noqa: E402
    estimate_one_for_everyone,
    estimate_per_person,
    simulate_people,
)


def _travel_error_mm(**params):
    s = simulate_session(head_gain=0.9, depth_jitter=0.15, **params)
    est = detect_compressions(s.t, s.y)
    travel = head_travel(s.t, s.y, est)
    ei, ti = pair_events(est, s.true_times)
    return (travel[ei] - 0.9 * s.true_depths_m[ti]) * 1000


def test_head_travel_clean_signal():
    err = _travel_error_mm(noise_m=0.0, sway_m=0.0)
    assert np.abs(err).max() < 1.5


def test_head_travel_with_noise_sway_and_pauses():
    err = _travel_error_mm(pauses=((17.0, 4.0), (38.0, 4.0)))
    assert not np.isnan(err).any()
    assert np.abs(err).mean() < 2.0


def test_head_travel_empty():
    assert head_travel([0, 1, 2], [0, 0, 0], []).size == 0


def test_fit_calibration_recovers_line():
    rng = np.random.default_rng(0)
    travel = rng.uniform(0.03, 0.07, 200)
    depth = 1.1 * travel + 0.002 + rng.normal(0, 0.0005, 200)
    cal = fit_calibration(travel, depth)
    assert cal.gain == pytest.approx(1.1, abs=0.02)
    assert cal.offset_m == pytest.approx(0.002, abs=0.001)


def test_fit_calibration_ignores_outliers():
    rng = np.random.default_rng(1)
    travel = rng.uniform(0.03, 0.07, 200)
    depth = 1.1 * travel + rng.normal(0, 0.0005, 200)
    depth[:6] += 0.03  # six bad pairs
    cal = fit_calibration(travel, depth)
    assert cal.gain == pytest.approx(1.1, abs=0.03)


def test_fit_calibration_through_origin():
    travel = np.linspace(0.03, 0.07, 50)
    cal = fit_calibration(travel, 1.2 * travel, through_origin=True)
    assert cal.gain == pytest.approx(1.2)
    assert cal.offset_m == 0.0


def test_fit_calibration_needs_data():
    with pytest.raises(ValueError):
        fit_calibration([0.05, 0.06], [0.05, 0.06])


def test_calibration_depth():
    assert Calibration(gain=2.0, offset_m=0.01).depth([0.02]) == pytest.approx([0.05])


def test_depth_band():
    assert list(depth_band([0.049, 0.05, 0.055, 0.06, 0.061])) == [-1, 0, 0, 0, 1]


def test_score_depth_counts():
    true = np.array([0.040, 0.055, 0.055, 0.065])
    est = np.array([0.052, 0.055, 0.048, 0.066])   # 1st: false ok; 3rd: wrong band
    s = score_depth(est, true)
    assert s.n == 4
    assert s.band_accuracy == pytest.approx(0.5)
    assert s.false_reassurance_fraction == pytest.approx(0.25)
    assert not s.passed and not s.guardrail_passed


def test_score_depth_margin_reduces_false_ok():
    true = np.array([0.049] * 10)
    est = np.array([0.0505] * 10)
    assert score_depth(est, true).false_reassurance_fraction == 1.0
    assert score_depth(est, true, say_good_margin_m=0.001).false_reassurance_fraction == 0.0


def test_per_person_beats_one_for_everyone_when_people_differ():
    people = simulate_people(8, gain_spread=0.12, seed=0)
    truth = np.concatenate([p.test_depth for p in people])
    one = score_depth(np.concatenate(estimate_one_for_everyone(people)), truth)
    per = score_depth(np.concatenate(estimate_per_person(people)), truth)
    assert per.mean_abs_error_mm < one.mean_abs_error_mm
    assert per.mean_abs_error_mm < 3.0
    assert abs(per.bias_mm) < 1.0
