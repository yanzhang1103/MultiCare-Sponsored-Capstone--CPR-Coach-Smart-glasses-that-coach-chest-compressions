import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine import (  # noqa: E402
    align_events,
    align_session,
    detect_compressions,
    score_rate,
    simulate_manikin_clock,
    simulate_session,
)


def _session(seed=0, **params):
    return simulate_session(seed=seed, start_cue=3, **params)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("offset", [-37.5, 0.0, 12.3])
def test_recovers_offset(seed, offset):
    s = _session(seed)
    manikin = simulate_manikin_clock(s.true_times, offset_s=offset, seed=seed)
    al = align_session(s.t, s.y, manikin)
    assert al.ok, al.reason
    assert np.abs(al.to_glasses_time(manikin) - s.true_times).mean() < 0.005
    assert al.paired_fraction > 0.95


def test_recovers_clock_drift():
    s = _session(1)
    manikin = simulate_manikin_clock(s.true_times, offset_s=5.0, drift_ppm=300, seed=1)
    al = align_session(s.t, s.y, manikin)
    assert al.ok, al.reason
    assert al.drift_ppm == pytest.approx(300, abs=60)


def test_large_clock_values():
    # e.g. the manikin stamps time in seconds since 1970
    s = _session(2)
    manikin = simulate_manikin_clock(s.true_times, offset_s=-1.79e9, seed=2)
    al = align_session(s.t, s.y, manikin)
    assert al.ok, al.reason
    assert np.abs(al.to_glasses_time(manikin) - s.true_times).mean() < 0.005


def test_time_conversion_round_trip():
    s = _session(3)
    manikin = simulate_manikin_clock(s.true_times, offset_s=8.0, drift_ppm=200, seed=3)
    al = align_session(s.t, s.y, manikin)
    assert al.to_manikin_time(al.to_glasses_time(manikin)) == pytest.approx(manikin)


def test_steady_rhythm_without_cue_is_flagged():
    s = simulate_session(seed=0, rate_jitter=0.005)
    manikin = simulate_manikin_clock(s.true_times, offset_s=4.0, seed=0)
    al = align_session(s.t, s.y, manikin)
    assert not al.ok
    assert "another alignment" in al.reason


def test_steady_rhythm_with_cue_is_accepted():
    s = _session(0, rate_jitter=0.005)
    manikin = simulate_manikin_clock(s.true_times, offset_s=4.0, seed=0)
    al = align_session(s.t, s.y, manikin)
    assert al.ok, al.reason
    assert np.abs(al.to_glasses_time(manikin) - s.true_times).mean() < 0.005


def test_unrelated_recordings_are_flagged():
    a = simulate_session(seed=0, start_cue=3)
    b = simulate_session(seed=7, rate_cpm=95, rate_jitter=0.08)
    al = align_session(a.t, a.y, b.true_times)
    assert not al.ok


def test_too_few_compressions():
    al = align_events([1.0, 1.5, 2.0], [1.0, 1.5, 2.0])
    assert not al.ok
    assert al.n_paired == 0


def test_scores_after_alignment_match_shared_clock():
    s = _session(4, rate_cpm=115, rate_drift_cpm=-20)
    manikin = simulate_manikin_clock(s.true_times, offset_s=-21.0, drift_ppm=150, seed=4)
    al = align_session(s.t, s.y, manikin)
    est = detect_compressions(s.t, s.y)
    rs = score_rate(est, al.to_glasses_time(manikin), s.duration_s)
    assert rs.passed and rs.n_within == rs.n_segments
