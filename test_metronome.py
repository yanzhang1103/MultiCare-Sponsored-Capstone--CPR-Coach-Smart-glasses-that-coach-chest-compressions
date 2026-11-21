import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine import Coach, CoachConfig, Compression, run_live, simulate_session  # noqa: E402
from live_eval import evaluate_live  # noqa: E402

PERSON = dict(head_gain=0.9, head_coupling_jitter=0.04)


def _drive(coach, rate_at, seconds, start=0.0, step=0.02):
    """Compress at rate_at(t) per minute, calling the coach every `step` seconds."""
    states = []
    t = start
    next_comp = start
    while t < start + seconds:
        if t >= next_comp:
            coach.on_compression(Compression(next_comp, 0.05, next_comp))
            next_comp += 60.0 / rate_at(next_comp)
        states.append(coach.tick(t))
        t += step
    return states


def _on_at(states, t):
    return min(states, key=lambda s: abs(s.t - t)).metronome_on


def test_ticks_before_the_first_compression():
    coach = Coach()
    states = [coach.tick(k * 0.02) for k in range(500)]      # ten seconds, nobody compressing
    assert all(s.metronome_on for s in states)
    assert sum(s.metronome_tick for s in states) == pytest.approx(110 / 6, abs=1)


def test_tick_rate_is_the_set_tempo():
    coach = Coach(config=CoachConfig(metronome_mode="always", metronome_bpm=110))
    states = _drive(coach, lambda t: 110, 60.0)
    assert sum(s.metronome_tick for s in states) == pytest.approx(110, abs=1)
    ticks = np.array([s.t for s in states if s.metronome_tick])
    assert np.diff(ticks) == pytest.approx(60 / 110, abs=0.021)


def test_adaptive_falls_silent_once_the_rate_holds():
    coach = Coach()
    states = _drive(coach, lambda t: 110, 60.0)
    assert _on_at(states, 5.0)
    assert not _on_at(states, 20.0)
    assert not _on_at(states, 59.0)
    off_at = next(s.t for s in states if not s.metronome_on)
    assert 10.0 < off_at < 16.0      # a few seconds for a first reading, then ten held


def test_adaptive_never_falls_silent_when_too_slow():
    coach = Coach()
    states = _drive(coach, lambda t: 85, 60.0)
    assert all(s.metronome_on for s in states)


def test_adaptive_comes_back_on_a_drift():
    coach = Coach()
    states = _drive(coach, lambda t: 110 if t < 40 else 90, 80.0)
    assert not _on_at(states, 38.0)
    assert _on_at(states, 52.0)
    back_at = next(s.t for s in states if s.t > 38.0 and s.metronome_on)
    assert 40.0 < back_at < 50.0


def test_adaptive_settles_again_after_recovering():
    coach = Coach()
    states = _drive(coach, lambda t: 110 if (t < 40 or t > 60) else 90, 110.0)
    assert _on_at(states, 55.0)
    assert not _on_at(states, 105.0)


def test_voice_waits_for_the_tick_then_speaks():
    coach = Coach()
    states = _drive(coach, lambda t: 110 if t < 40 else 90, 80.0)
    back_at = next(s.t for s in states if s.t > 38.0 and s.metronome_on)
    said = [s.t for s in states if s.prompt == "A little faster"]
    assert said, "still slow after the tick came back, so the voice should speak"
    assert said[0] - back_at >= CoachConfig().metronome_grace_s - 0.1


def test_no_rate_words_if_the_tick_fixes_it():
    coach = Coach()
    # Drifts slow for five seconds, then recovers once the tick is back.
    states = _drive(coach, lambda t: 90 if 40 <= t < 47 else 110, 80.0)
    assert not any(s.prompt in ("A little faster", "A little slower") for s in states)


def test_short_pause_does_not_restart_the_tick():
    coach = Coach()
    states = _drive(coach, lambda t: 110, 30.0)
    assert not states[-1].metronome_on
    # Four seconds off for breaths, then on again at the same rate.
    for k in range(200):
        states.append(coach.tick(30.0 + k * 0.02))
    states += _drive(coach, lambda t: 110, 20.0, start=34.0)
    assert not any(s.metronome_on for s in states if s.t > 30.0)


def test_long_stop_silences_then_restarts():
    coach = Coach(config=CoachConfig(metronome_mode="always"))
    states = _drive(coach, lambda t: 110, 20.0)
    for k in range(1000):                              # twenty seconds with hands off
        states.append(coach.tick(20.0 + k * 0.02))
    assert _on_at(states, 25.0)                        # still ticking through a short stop
    assert not _on_at(states, 35.0)                    # silent after ten seconds
    states += _drive(coach, lambda t: 110, 5.0, start=40.0)
    assert states[-1].metronome_on


def test_off_mode_never_ticks():
    coach = Coach(config=CoachConfig(metronome_mode="off"))
    states = _drive(coach, lambda t: 85, 30.0)
    assert not any(s.metronome_on or s.metronome_tick for s in states)


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        Coach(config=CoachConfig(metronome_mode="sometimes"))


def test_on_a_simulated_session():
    sim = simulate_session(seed=0, rate_cpm=110, depth_m=0.055, **PERSON)
    r = evaluate_live(sim, None)
    assert r.metronome_on < 0.25
    sim = simulate_session(seed=0, rate_cpm=115, rate_drift_cpm=-20, depth_m=0.055, **PERSON)
    r = evaluate_live(sim, None)
    assert len(r.metronome_spans) >= 2                 # on at the start, back when the rate sagged
    assert r.metronome_spans[-1][0] > 60.0
    _, states = run_live(sim.t, sim.y, None, CoachConfig(metronome_mode="always"))
    assert np.mean([s.metronome_on for s in states]) > 0.95
