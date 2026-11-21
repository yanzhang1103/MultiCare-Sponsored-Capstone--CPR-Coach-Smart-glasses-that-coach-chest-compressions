r"""Write simulated sessions in the real file format.

    python make_sample_sessions.py
    python make_sample_sessions.py --bench     # also a 25-session bench set, 5 people

Creates sessions\sample_P1_sweep (a calibration sweep), sessions\sample_P1_test
and sessions\sample_P1_pillow (no manikin file). They show exactly what the
capture app and the manikin converter have to produce, and they let the
whole pipeline run before any device arrives.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cpr_engine import save_session, simulate_manikin_clock, simulate_session

HEAD_HEIGHT_M = 0.75      # head height above the floor when kneeling
HEAD_GAIN = 0.9


def write(folder: Path, sim, meta: dict, with_manikin: bool, offset_s: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = sim.t.size
    position = np.column_stack(
        (
            0.004 * rng.standard_normal(n).cumsum() / np.sqrt(n),   # x: slow wander
            HEAD_HEIGHT_M + sim.y,                                  # y: up
            0.004 * rng.standard_normal(n).cumsum() / np.sqrt(n),   # z: slow wander
        )
    )
    rotation = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))                # looking straight ahead
    manikin_t = manikin_depth = None
    if with_manikin:
        manikin_t = simulate_manikin_clock(
            sim.true_times, offset_s=offset_s, drift_ppm=60.0, seed=seed
        )
        manikin_depth = sim.true_depths_m
    save_session(folder, meta, sim.t, position, rotation, manikin_t, manikin_depth)
    print(f"wrote {folder}")


def write_bench(root: Path, n_people: int = 5, gain_spread: float = 0.06) -> None:
    """A bench set: each person records one calibration sweep and four test sessions."""
    rng = np.random.default_rng(2026)
    gains = HEAD_GAIN * (1.0 + gain_spread * rng.standard_normal(n_people))
    tests = [
        dict(rate_cpm=112, depth_m=0.055),
        dict(rate_cpm=92, depth_m=0.043),
        dict(rate_cpm=128, depth_m=0.064),
        dict(rate_cpm=116, rate_drift_cpm=-14, depth_m=0.050, pauses=((30.0, 4.0),)),
    ]
    for i, gain in enumerate(gains):
        person = f"P{i + 1}"
        person_args = dict(head_gain=float(gain), head_coupling_jitter=0.04, start_cue=3)
        common = dict(person=person, device="simulated", up_axis="y", surface="manikin")
        seed = 3000 + 100 * i
        sweep = simulate_session(duration_s=75, depth_m=0.035, depth_drift_m=0.035,
                                 depth_jitter=0.05, seed=seed, **person_args)
        sid = f"bench_{person}_sweep"
        write(root / sid, sweep, dict(session_id=sid, kind="calibration", **common),
              with_manikin=True, offset_s=float(rng.uniform(-30, 30)), seed=seed)
        for k, params in enumerate(tests):
            sim = simulate_session(duration_s=90, seed=seed + 1 + k, **person_args, **params)
            sid = f"bench_{person}_test{k + 1}"
            write(root / sid, sim, dict(session_id=sid, kind="test", **common),
                  with_manikin=True, offset_s=float(rng.uniform(-30, 30)), seed=seed + 1 + k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="also write sessions\\bench")
    args = parser.parse_args()

    root = Path("sessions")
    common = dict(person="P1", device="simulated", up_axis="y",
                  recorded_at="2026-11-16T15:00:00-08:00")

    sweep = simulate_session(
        duration_s=90, start_cue=3, depth_m=0.035, depth_drift_m=0.035, depth_jitter=0.05,
        head_gain=HEAD_GAIN, head_coupling_jitter=0.04, seed=11,
    )
    write(root / "sample_P1_sweep", sweep,
          dict(session_id="sample_P1_sweep", surface="manikin", kind="calibration",
               notes="depth from shallow to deep", **common),
          with_manikin=True, offset_s=7.3, seed=11)

    test = simulate_session(
        duration_s=120, start_cue=3, rate_cpm=118, rate_drift_cpm=-16, depth_m=0.054,
        pauses=((45.0, 4.0), (85.0, 4.0)), head_gain=HEAD_GAIN, head_coupling_jitter=0.04,
        n_glances=4, seed=12,
    )
    write(root / "sample_P1_test", test,
          dict(session_id="sample_P1_test", surface="manikin", kind="test", **common),
          with_manikin=True, offset_s=-12.8, seed=12)

    pillow = simulate_session(
        duration_s=60, rate_cpm=104, depth_m=0.046, head_gain=HEAD_GAIN,
        head_coupling_jitter=0.04, seed=13,
    )
    write(root / "sample_P1_pillow", pillow,
          dict(session_id="sample_P1_pillow", surface="pillow", kind="practice", **common),
          with_manikin=False, offset_s=0.0, seed=13)

    if args.bench:
        write_bench(root / "bench")


if __name__ == "__main__":
    main()
