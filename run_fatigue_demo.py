r"""Run the fatigue alert on simulated four-minute sessions and print how it did.

    python run_fatigue_demo.py
    python run_fatigue_demo.py --runs 30
    python run_fatigue_demo.py --show "fades 25% after the first minute"
"""

from __future__ import annotations

import argparse

import numpy as np

from cpr_engine import simulate_session
from live_eval import evaluate_fatigue, simulated_calibration

_PERSON = dict(head_gain=0.9, head_coupling_jitter=0.04, rate_cpm=110, duration_s=240)
_BREATHS = tuple((float(s), 4.0) for s in range(17, 230, 21))

CASES = {
    "steady, no fade": dict(depth_m=0.055),
    "fades 8% (under the limit)": dict(depth_m=0.056, depth_drift_m=-0.0045, depth_drift_from_s=60),
    "fades 25% after the first minute": dict(depth_m=0.056, depth_drift_m=-0.014, depth_drift_from_s=60),
    "fades 40% after 45 seconds": dict(depth_m=0.056, depth_drift_m=-0.022, depth_drift_from_s=45),
    "fades 25%, pauses for breaths": dict(
        depth_m=0.056, depth_drift_m=-0.014, depth_drift_from_s=60, pauses=_BREATHS),
    "steady, pauses for breaths": dict(depth_m=0.055, pauses=_BREATHS),
    "steady, uneven depth": dict(depth_m=0.055, depth_jitter=0.25),
    "steady, noisy sensor, glances": dict(depth_m=0.055, noise_m=0.005, sway_m=0.03, n_glances=20),
}


def _time(seconds: float) -> str:
    if np.isnan(seconds):
        return "-"
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--show", help="print one session's spoken prompts")
    args = parser.parse_args()

    calibration = simulated_calibration()

    if args.show:
        r = evaluate_fatigue(simulate_session(seed=0, **_PERSON, **CASES[args.show]), calibration)
        print(f"Spoken prompts, {args.show}:")
        for t, words in r.prompts:
            print(f"  {_time(t):>5}   {words}")
        return

    header = (
        f"{'case':<34}{'true fade':>10}{'measured':>10}{'alerted':>9}"
        f"{'true 15%':>10}{'alert at':>10}{'swap at':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, params in CASES.items():
        rs = [
            evaluate_fatigue(simulate_session(seed=seed, **_PERSON, **params), calibration)
            for seed in range(args.runs)
        ]
        alerted = [r for r in rs if not np.isnan(r.alert_s)]
        true_fade = np.nanmedian([r.true_fade_s for r in rs]) if any(
            not np.isnan(r.true_fade_s) for r in rs) else float("nan")
        alert_at = float(np.median([r.alert_s for r in alerted])) if alerted else float("nan")
        swaps = ", ".join(_time(t) for t in rs[0].swap_s) or "-"
        print(
            f"{name:<34}{np.mean([r.true_decline for r in rs]):>10.0%}"
            f"{np.nanmean([r.measured_decline for r in rs]):>10.0%}"
            f"{f'{len(alerted)}/{len(rs)}':>9}{_time(true_fade):>10}{_time(alert_at):>10}{swaps:>12}"
        )

    print()
    print(f"Each case: {args.runs} four-minute sessions at 110 per minute.")
    print("true fade: how much shallower the true depth was at the end than in the first 30 seconds")
    print("measured:  the same, as the monitor read it from head travel")
    print('alerted:   sessions where "You\'re tiring" was spoken')
    print("true 15%:  when the true depth had fallen 15%; alert at: when the prompt came")
    print("swap at:   when the two-minute swap reminder came (first session)")


if __name__ == "__main__":
    main()
