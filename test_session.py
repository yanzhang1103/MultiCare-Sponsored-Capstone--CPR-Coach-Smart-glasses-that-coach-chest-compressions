import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine import (  # noqa: E402
    SessionError,
    fit_calibration,
    format_scorecard,
    load_calibration,
    load_session,
    pair_session,
    save_calibration,
    save_session,
    score_session,
    simulate_manikin_clock,
    simulate_session,
)


def _write(folder, sim, with_manikin=True, offset=7.3, meta=None, depth_scale=1.0,
           height_scale=1.0, time_scale=1.0):
    n = sim.t.size
    pos = np.column_stack((np.zeros(n), (0.75 + sim.y) * height_scale, np.zeros(n)))
    manikin_t = manikin_depth = None
    if with_manikin:
        manikin_t = simulate_manikin_clock(sim.true_times, offset_s=offset, seed=1)
        manikin_depth = sim.true_depths_m * depth_scale
    save_session(folder, meta or {"person": "P1"}, sim.t * time_scale, pos, None,
                 manikin_t, manikin_depth)
    return folder


def _sim(seed=0, **params):
    base = dict(start_cue=3, head_coupling_jitter=0.03)
    base.update(params)
    return simulate_session(seed=seed, **base)


def test_round_trip(tmp_path):
    sim = _sim()
    s = load_session(_write(tmp_path / "s1", sim))
    assert s.session_id == "s1"
    assert s.meta["format_version"] == 1
    assert s.t.size == sim.t.size
    assert np.allclose(s.height, 0.75 + sim.y, atol=1e-5)
    assert s.manikin_t.size == sim.true_times.size
    assert not s.warnings


def test_scorecard_end_to_end(tmp_path):
    sweep = _sim(1, duration_s=90, depth_m=0.035, depth_drift_m=0.035, depth_jitter=0.05)
    test = _sim(2, depth_m=0.066, pauses=((45.0, 4.0),))
    sweep_s = load_session(_write(tmp_path / "sweep", sweep, offset=3.0))
    test_s = load_session(_write(tmp_path / "test", test, offset=-20.0))

    p = pair_session(sweep_s)
    assert p.alignment.ok
    cal = fit_calibration(p.travel_m[p.est_idx], p.true_depth_m[p.true_idx])
    save_calibration(tmp_path / "cal.json", cal, {"fitted_from": ["sweep"]})
    cal = load_calibration(tmp_path / "cal.json")

    score = score_session(test_s, cal)
    assert score.alignment.ok
    assert score.alignment.offset_s == pytest.approx(-20.0, abs=0.01)
    assert score.events.n_missed == 0 and score.events.n_extra == 0
    assert score.rate.passed and score.rate.guardrail_passed
    assert score.pauses.n_found == 1 and score.pauses.n_false == 0
    assert score.depth.mean_abs_error_mm < 3.5
    assert score.travel_vs_depth.correlation > 0.5
    text = format_scorecard(score)
    assert "PASS" in text and "alignment" in text


def test_no_calibration_means_no_depth_score(tmp_path):
    score = score_session(load_session(_write(tmp_path / "s", _sim())))
    assert score.depth is None
    assert score.travel_vs_depth is not None
    assert "no calibration given" in format_scorecard(score)


def test_pillow_session_has_measurements_only(tmp_path):
    sim = _sim(start_cue=0, rate_cpm=104)
    score = score_session(load_session(_write(tmp_path / "p", sim, with_manikin=False)))
    assert score.alignment is None
    assert score.measurements.n_compressions == pytest.approx(sim.true_times.size, abs=1)
    assert score.measurements.rate_cpm == pytest.approx(104, abs=2)
    assert "No manikin file" in format_scorecard(score)


def test_flagged_alignment_skips_depth(tmp_path):
    sim = simulate_session(seed=0, rate_jitter=0.005)   # steady rhythm, no cue
    score = score_session(load_session(_write(tmp_path / "s", sim)))
    assert not score.alignment.ok
    assert score.depth is None and score.travel_vs_depth is None
    assert "FLAGGED" in format_scorecard(score)


def test_missing_pose_file(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SessionError, match="missing file: head_pose.csv"):
        load_session(tmp_path / "empty")


def test_missing_column(tmp_path):
    folder = _write(tmp_path / "s", _sim())
    (folder / "session.json").write_text(json.dumps({"up_axis": "w"}))
    with pytest.raises(SessionError, match="up_axis"):
        load_session(folder)
    text = (folder / "head_pose.csv").read_text().replace("y_m", "height")
    (folder / "head_pose.csv").write_text(text)
    (folder / "session.json").write_text(json.dumps({"up_axis": "y"}))
    with pytest.raises(SessionError, match="no column named y_m"):
        load_session(folder)


def test_depth_in_centimetres_is_rejected(tmp_path):
    folder = _write(tmp_path / "s", _sim(), depth_scale=100.0)
    with pytest.raises(SessionError, match="metres"):
        load_session(folder)


def test_height_in_centimetres_is_rejected(tmp_path):
    folder = _write(tmp_path / "s", _sim(), height_scale=100.0)
    with pytest.raises(SessionError, match="metres"):
        load_session(folder)


def test_time_in_milliseconds_is_rejected(tmp_path):
    folder = _write(tmp_path / "s", _sim(), time_scale=1000.0)
    with pytest.raises(SessionError, match="milliseconds"):
        load_session(folder)


def test_wrong_up_axis_warns(tmp_path):
    folder = _write(tmp_path / "s", _sim(), meta={"up_axis": "x"})
    s = load_session(folder)
    assert any("barely changes" in w for w in s.warnings)


def test_other_up_axis(tmp_path):
    sim = _sim()
    n = sim.t.size
    pos = np.column_stack((np.zeros(n), np.zeros(n), 0.75 + sim.y))
    save_session(tmp_path / "z", {"up_axis": "z"}, sim.t, pos)
    s = load_session(tmp_path / "z")
    assert np.allclose(s.height, 0.75 + sim.y, atol=1e-5)
