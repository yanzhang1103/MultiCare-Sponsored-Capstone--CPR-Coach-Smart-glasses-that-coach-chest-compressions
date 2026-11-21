"""Run the alignment on simulated sessions and print how often it is right.

    python run_align_demo.py
    python run_align_demo.py --runs 200

Each run gives the glasses and the manikin different clocks (a random
offset of up to 40 seconds), aligns them, and checks the answer.
"""

from __future__ import annotations

import argparse

import numpy as np

from cpr_engine import align_session, simulate_manikin_clock, simulate_session

_PAUSES = tuple((s, 4.0) for s in (17.0, 38.0, 59.0, 80.0, 101.0))
_STEADY = dict(rate_jitter=0.005)   # a rhythm as even as a metronome

# name: (glasses-side settings, manikin-side settings)
CASES = {
    "start cue, normal rhythm": (dict(start_cue=3), dict()),
    "start cue, very steady rhythm": (dict(start_cue=3, **_STEADY), dict()),
    "start cue, clocks drift apart": (dict(start_cue=3), dict(drift_ppm=300)),
    "start cue, noisy sensor, glances": (
        dict(start_cue=3, noise_m=0.005, sway_m=0.03, n_glances=10), dict()),
    "start cue, manikin misses 5%": (dict(start_cue=3), dict(drop_fraction=0.05)),
    "no cue, normal rhythm": (dict(), dict()),
    "no cue, very steady, with pauses": (dict(pauses=_PAUSES, **_STEADY), dict()),
    "no cue, very steady, no pauses": (dict(**_STEADY), dict()),
    "cue missed, normal rhythm": (dict(start_cue=3), dict(starts_at_s=10.0)),
    "cue missed, very steady rhythm": (dict(start_cue=3, **_STEADY), dict(starts_at_s=10.0)),
}

RIGHT_WITHIN_S = 0.05   # an alignment within 50 ms counts as right


def run_case(sim_params: dict, manikin_params: dict, runs: int):
    used_right = flagged = used_wrong = 0
    errors_ms = []
    for seed in range(runs):
        offset = float(np.random.default_rng(seed + 99).uniform(-40.0, 40.0))
        s = simulate_session(seed=seed, **sim_params)
        manikin = simulate_manikin_clock(
            s.true_times, offset_s=offset, seed=seed, **manikin_params
        )
        al = align_session(s.t, s.y, manikin)

        # Where does the alignment put the middle of the session, against the truth?
        drift = manikin_params.get("drift_ppm", 0.0) * 1e-6
        mid_glasses = s.duration_s / 2.0
        mid_manikin = (mid_glasses - offset) / (1.0 + drift)
        error = float(al.to_glasses_time(mid_manikin) - mid_glasses) if al.n_paired else np.nan
        right = abs(error) < RIGHT_WITHIN_S

        if not al.ok:
            flagged += 1
        elif right:
            used_right += 1
            errors_ms.append(abs(error) * 1000.0)
        else:
            used_wrong += 1
    typical = float(np.median(errors_ms)) if errors_ms else float("nan")
    return used_right, flagged, used_wrong, typical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    header = (
        f"{'case':<36}{'accepted, right':>17}{'flagged':>9}"
        f"{'accepted, wrong':>17}{'error ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, (sim_params, manikin_params) in CASES.items():
        used_right, flagged, used_wrong, typical = run_case(sim_params, manikin_params, args.runs)
        err = f"{typical:.1f}" if not np.isnan(typical) else "-"
        print(f"{name:<36}{used_right:>17}{flagged:>9}{used_wrong:>17}{err:>10}")

    print()
    print(f"Each case: {args.runs} runs.")
    print("accepted, right: the alignment passed its own checks and was within 50 ms")
    print("flagged:         the alignment failed its own checks, so it would not be used")
    print("accepted, wrong: passed its checks but was wrong. This must stay at zero")
    print("error ms:        typical error of the accepted alignments")


if __name__ == "__main__":
    main()
