import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine import (  # noqa: E402
    PROMPTS,
    Calibration,
    Coach,
    CoachConfig,
    Compression,
    FatigueConfig,
    FatigueMonitor,
    simulate_session,
)
from live_eval import evaluate_fatigue, simulated_calibration  # noqa: E402

PERSON = dict(head_gain=0.9, head_coupling_jitter=0.04, rate_cpm=110, duration_s=240)
PERIOD = 60.0 / 110.0


def _feed(monitor, travel_at, start=0.0, seconds=120.0):
    """Feed steady compressions; travel_at(t) gives the head travel at time t."""
    t = start
    while t < start + seconds:
        monitor.on_compression(t, travel_at(t))
        t += PERIOD
    return t


# ---- the monitor on its own ---------------------------------------------------


def test_no_reading_until_the_first_30_seconds_are_in():
    m = FatigueMonitor()
    _feed(m, lambda t: 0.05, seconds=25.0)
    assert np.isnan(m.baseline) and np.isnan(m.decline) and not m.tiring
    _feed(m, lambda t: 0.05, start=25.0, seconds=25.0)
    assert m.baseline == pytest.approx(0.05)
    assert m.decline == pytest.approx(0.0, abs=1e-9)


def test_steady_compressions_are_not_tiring():
    m = FatigueMonitor()
    rng = np.random.default_rng(0)
    _feed(m, lambda t: 0.05 * (1 + 0.08 * rng.standard_normal()), seconds=240.0)
    assert not m.tiring
    assert abs(m.decline) < 0.08


def test_a_fade_is_flagged_and_measured():
    m = FatigueMonitor()
    _feed(m, lambda t: 0.05 if t < 60 else 0.05 * (1 - 0.25 * min(1.0, (t - 60) / 60)), seconds=200.0)
    assert m.tiring
    assert m.decline == pytest.approx(0.25, abs=0.01)


def test_a_small_fade_is_not_flagged():
    m = FatigueMonitor()
    _feed(m, lambda t: 0.05 if t < 60 else 0.046, seconds=200.0)
    assert not m.tiring
    assert m.decline == pytest.approx(0.08, abs=0.01)


def test_a_brief_dip_is_not_flagged():
    m = FatigueMonitor()
    _feed(m, lambda t: 0.035 if 80 < t < 84 else 0.05, seconds=200.0)   # a few shallow ones
    assert not m.tiring


def test_recovery_clears_the_flag():
    m = FatigueMonitor()
    _feed(m, lambda t: 0.05 if (t < 60 or t > 130) else 0.038, seconds=200.0)
    assert not m.tiring


def test_swap_reminder_every_two_minutes():
    m = FatigueMonitor()
    due_at = []
    t = 0.0
    while t < 300.0:
        m.on_compression(t, 0.05)
        if m.swap_due:
            due_at.append(t)
            m.swap_said()
        t += PERIOD
    assert len(due_at) == 2
    assert due_at[0] == pytest.approx(120.0, abs=1.0)
    assert due_at[1] == pytest.approx(240.0, abs=1.5)


def test_short_pauses_count_toward_the_two_minutes_but_not_the_baseline():
    m = FatigueMonitor()
    t = 0.0
    first_due = None
    while t < 200.0 and first_due is None:
        for _ in range(30):                 # 30 compressions, then 4 seconds for breaths
            m.on_compression(t, 0.05)
            if m.swap_due and first_due is None:
                first_due = t
            t += PERIOD
        t += 4.0
    assert first_due == pytest.approx(120.0, abs=5.0)
    assert m.active_s < m.elapsed_s


def test_a_long_rest_starts_again():
    m = FatigueMonitor()
    t = _feed(m, lambda t: 0.05 if t < 60 else 0.036, seconds=150.0)
    assert m.tiring
    m.on_compression(t + 45.0, 0.036)       # back after a 45-second rest
    assert not m.tiring and np.isnan(m.baseline) and m.elapsed_s == 0.0


def test_rebaseline_measures_afresh():
    m = FatigueMonitor()
    t = _feed(m, lambda t: 0.065, seconds=50.0)
    m.rebaseline()
    _feed(m, lambda t: 0.052, start=t, seconds=100.0)     # eased off on request
    assert m.baseline == pytest.approx(0.052)
    assert not m.tiring


def test_custom_threshold():
    m = FatigueMonitor(FatigueConfig(decline=0.05, recover=0.02))
    _feed(m, lambda t: 0.05 if t < 60 else 0.046, seconds=200.0)
    assert m.tiring


# ---- through the coach ----------------------------------------------------------


@pytest.fixture(scope="module")
def calibration():
    return simulated_calibration()


def test_coach_says_tiring_when_depth_fades(calibration):
    sim = simulate_session(seed=0, depth_m=0.056, depth_drift_m=-0.014, depth_drift_from_s=60, **PERSON)
    r = evaluate_fatigue(sim, calibration)
    assert not np.isnan(r.alert_s)
    assert r.alert_s > r.true_fade_s - 30.0          # not long before the true fade
    assert r.alert_s < r.true_fade_s + 45.0          # and not long after
    assert r.measured_decline == pytest.approx(r.true_decline, abs=0.05)


def test_coach_stays_quiet_when_steady(calibration):
    for seed in range(4):
        r = evaluate_fatigue(simulate_session(seed=seed, depth_m=0.055, **PERSON), calibration)
        assert np.isnan(r.alert_s)
        assert len(r.swap_s) == 1 and r.swap_s[0] == pytest.approx(121.0, abs=3.0)


def test_fatigue_works_without_a_calibration():
    sim = simulate_session(seed=1, depth_m=0.056, depth_drift_m=-0.016, depth_drift_from_s=60, **PERSON)
    r = evaluate_fatigue(sim, None)
    assert not np.isnan(r.alert_s)


def test_fatigue_can_be_switched_off(calibration):
    sim = simulate_session(seed=0, depth_m=0.056, depth_drift_m=-0.016, depth_drift_from_s=60, **PERSON)
    r = evaluate_fatigue(sim, calibration, CoachConfig(fatigue_enabled=False))
    assert np.isnan(r.alert_s) and r.swap_s == []


def test_easing_off_when_told_is_not_fatigue():
    # Too deep at first; the coach says "Not so deep"; the person eases off into range.
    coach = Coach(Calibration(gain=1.0, offset_m=0.0))
    said = []
    t = 0.0
    eased = False
    while t < 200.0:
        travel = 0.056 if eased else 0.070
        coach.on_compression(Compression(t, travel, t))
        s = coach.tick(t + 0.01)
        if s.prompt:
            said.append(s.prompt)
            if s.prompt == PROMPTS["ease off"]:
                eased = True
        assert not s.tiring
        t += PERIOD
    assert PROMPTS["ease off"] in said
    assert PROMPTS["tiring"] not in said


def test_a_correction_is_not_repeated_too_soon(calibration):
    # Hovering at the shallow edge makes the cue come and go; the words must not.
    sim = simulate_session(seed=0, depth_m=0.049, **PERSON)
    r = evaluate_fatigue(sim, calibration)
    times = [t for t, w in r.prompts if w == PROMPTS["push harder"]]
    assert len(times) >= 2
    assert np.diff(times).min() >= CoachConfig().repeat_s - 0.1
