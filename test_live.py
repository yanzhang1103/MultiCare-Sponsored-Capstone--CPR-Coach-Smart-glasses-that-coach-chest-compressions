import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine import (  # noqa: E402
    Coach,
    CoachConfig,
    Compression,
    StreamingDetector,
    detect_compressions,
    head_travel,
    match_events,
    pair_events,
    run_live,
    score_pauses,
    score_rate,
    simulate_session,
)
from cpr_engine.coach import _Cue  # noqa: E402
from live_eval import evaluate_live, simulated_calibration  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

PERSON = dict(head_gain=0.9, head_coupling_jitter=0.04)


def _stream(sim):
    det = StreamingDetector()
    out = []
    for t, y in zip(sim.t.tolist(), sim.y.tolist()):
        out += det.update(t, y)
    return out


# ---- live detector ---------------------------------------------------------------


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_live_detector_meets_rate_pass_mark(name, seed):
    sim = simulate_session(seed=seed, **SCENARIOS[name])
    est = np.array([c.t for c in _stream(sim)])
    rs = score_rate(est, sim.true_times, sim.duration_s)
    ps = score_pauses(est, sim.true_times)
    assert rs.passed, f"{rs.n_within}/{rs.n_segments}"
    assert ps.n_missed == 0 and ps.n_false == 0


def test_live_matches_offline_on_a_clean_session():
    sim = simulate_session(seed=0)
    live = np.array([c.t for c in _stream(sim)])
    offline = detect_compressions(sim.t, sim.y)
    m = match_events(live, offline, tolerance_s=0.05)
    assert m.n_missed == 0 and m.n_extra == 0


def test_delay_is_short():
    sim = simulate_session(seed=0)
    comps = _stream(sim)
    ei, ti = pair_events(np.array([c.t for c in comps]), sim.true_times)
    delays = np.array([comps[i].detected_at - sim.true_times[j] for i, j in zip(ei, ti)])
    assert np.median(delays) < 0.2
    assert np.all(delays > 0)


def test_no_compressions_from_glances_alone():
    for seed in range(5):
        sim = simulate_session(seed=seed, pauses=((0.0, 200.0),), n_glances=8)
        assert _stream(sim) == []


def test_no_extra_compression_after_stopping():
    sim = simulate_session(seed=0, pauses=((20.0, 200.0),))
    comps = _stream(sim)
    assert len(comps) == sim.true_times.size


def test_steadied_travel_is_as_good_as_offline():
    sim = simulate_session(seed=0, head_gain=0.9, depth_jitter=0.15)
    comps = _stream(sim)
    travel = np.array([c.travel_m for c in comps])
    for k in range(len(comps) - 1):
        if not np.isnan(comps[k + 1].prev_rise_m):
            travel[k] = 0.5 * (travel[k] + comps[k + 1].prev_rise_m)
    ei, ti = pair_events(np.array([c.t for c in comps]), sim.true_times)
    live_err = np.abs(travel[ei] - 0.9 * sim.true_depths_m[ti]).mean()

    off = detect_compressions(sim.t, sim.y)
    oi, oj = pair_events(off, sim.true_times)
    off_err = np.abs(head_travel(sim.t, sim.y, off)[oi] - 0.9 * sim.true_depths_m[oj]).mean()
    assert live_err < off_err + 0.0005


def test_dropout_restarts_and_recovers():
    sim = simulate_session(seed=0)
    keep = (sim.t < 40.0) | (sim.t > 43.0)      # three seconds of samples lost
    det = StreamingDetector()
    out = []
    for t, y in zip(sim.t[keep].tolist(), sim.y[keep].tolist()):
        out += det.update(t, y)
    times = np.array([c.t for c in out])
    assert np.count_nonzero(times > 50.0) > 100
    assert not np.any((times > 40.5) & (times < 43.0))


def test_filter_coefficients_for_porting():
    coeffs = StreamingDetector().filter_coefficients()
    assert len(coeffs["band_pass_sections_b0_b1_b2_a1_a2"]) == 2
    assert len(coeffs["low_pass_sections_b0_b1_b2_a1_a2"]) == 1
    assert all(len(row) == 5 for row in coeffs["band_pass_sections_b0_b1_b2_a1_a2"])


# ---- coach -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def calibration():
    return simulated_calibration()


def _words(result):
    return [w for _, w in result.prompts]


def test_too_slow_is_told_faster(calibration):
    r = evaluate_live(simulate_session(seed=0, rate_cpm=85, depth_m=0.055, **PERSON), calibration)
    assert "A little faster" in _words(r)
    assert "A little slower" not in _words(r)
    # With the metronome ticking, the voice waits a few seconds for the tick to work.
    assert 7.0 < r.first_correction_s < 12.0
    assert r.rate_false_good == 0.0 and r.rate_wrong_way == 0.0
    assert r.rate_right > 0.9


def test_too_slow_without_metronome_is_told_sooner(calibration):
    sim = simulate_session(seed=0, rate_cpm=85, depth_m=0.055, **PERSON)
    r = evaluate_live(sim, calibration, CoachConfig(metronome_mode="off"))
    assert "A little faster" in _words(r)
    assert r.first_correction_s < 7.0
    assert r.metronome_on == 0.0


def test_too_fast_is_told_slower(calibration):
    r = evaluate_live(simulate_session(seed=0, rate_cpm=140, depth_m=0.055, **PERSON), calibration)
    assert "A little slower" in _words(r)
    assert "A little faster" not in _words(r)
    assert r.rate_false_good == 0.0


def test_too_shallow_is_told_push_harder(calibration):
    r = evaluate_live(simulate_session(seed=0, rate_cpm=110, depth_m=0.042, **PERSON), calibration)
    assert "Push harder" in _words(r)
    assert "Not so deep" not in _words(r)
    assert r.depth_false_good < 0.01
    assert r.depth_right > 0.9


def test_too_deep_is_told_not_so_deep(calibration):
    r = evaluate_live(simulate_session(seed=0, rate_cpm=110, depth_m=0.068, **PERSON), calibration)
    assert "Not so deep" in _words(r)
    assert "Push harder" not in _words(r)


def test_on_target_is_left_alone_about_rate(calibration):
    r = evaluate_live(simulate_session(seed=0, rate_cpm=110, depth_m=0.055, **PERSON), calibration)
    assert "A little faster" not in _words(r) and "A little slower" not in _words(r)
    assert r.rate_right > 0.9
    assert r.prompts_per_min < 2.0


@pytest.mark.parametrize("seed", range(4))
def test_false_good_stays_under_the_guardrail(calibration, seed):
    for params in (dict(rate_cpm=97, depth_m=0.048), dict(rate_cpm=123, depth_m=0.062)):
        r = evaluate_live(simulate_session(seed=seed, **params, **PERSON), calibration)
        assert r.rate_false_good < 0.05
        assert r.depth_false_good < 0.05


def test_no_calibration_means_no_depth_cue():
    sim = simulate_session(seed=0, rate_cpm=110, depth_m=0.03, **PERSON)
    _, states = run_live(sim.t, sim.y, None)
    assert all(s.depth_cue == "" for s in states)
    assert all(np.isnan(s.depth_m) for s in states)
    assert not any(s.prompt in ("Push harder", "Not so deep") for s in states)


def test_prompts_are_spaced_out(calibration):
    sim = simulate_session(seed=0, rate_cpm=88, depth_m=0.042, **PERSON)
    r = evaluate_live(sim, calibration)
    times = np.array([t for t, _ in r.prompts])
    assert len(times) > 4
    assert np.diff(times).min() >= CoachConfig().prompt_gap_s - 0.05


def test_hands_off_prompt_once_per_pause(calibration):
    sim = simulate_session(seed=0, rate_cpm=110, depth_m=0.055, pauses=((30.0, 25.0),), **PERSON)
    r = evaluate_live(sim, calibration)
    restart = [t for t, w in r.prompts if w == "Start compressions again"]
    assert len(restart) == 1
    assert 39.0 < restart[0] < 42.0


def test_no_hands_off_prompt_before_the_first_compression():
    coach = Coach()
    for k in range(300):
        assert coach.tick(k * 0.1).prompt is None


def test_coach_resets_after_a_pause():
    coach = Coach()
    t = 0.0
    for _ in range(12):                     # 120 per minute
        coach.on_compression(Compression(t, 0.05, t))
        t += 0.5
    assert coach.tick(t).rate_cpm == pytest.approx(120.0)
    t += 5.0                                # a pause
    assert coach.tick(t).active is False
    for _ in range(3):                      # too few for a rate yet
        coach.on_compression(Compression(t, 0.05, t))
        t += 0.75
    assert np.isnan(coach.tick(t - 0.7).rate_cpm)


def test_cue_must_hold_before_it_shows():
    cue = _Cue()
    assert cue.update("faster", 0.0, 1.5) == ""
    assert cue.update("faster", 1.0, 1.5) == ""
    assert cue.update("faster", 1.6, 1.5) == "faster"


def test_good_is_withdrawn_at_once():
    cue = _Cue()
    cue.update("good", 0.0, 1.5)
    assert cue.update("good", 2.0, 1.5) == "good"
    assert cue.update("", 2.1, 1.5) == ""
    cue.update("good", 3.0, 1.5)
    cue.update("good", 5.0, 1.5)
    assert cue.update("faster", 5.1, 1.5) == ""
