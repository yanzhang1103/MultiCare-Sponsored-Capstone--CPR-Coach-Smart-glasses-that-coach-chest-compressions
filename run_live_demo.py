r"""Run the live detector and coach on simulated sessions and print how they did.

    python run_live_demo.py
    python run_live_demo.py --runs 20
    python run_live_demo.py --show "slow and shallow"     # print one session's spoken prompts
    python run_live_demo.py --metronome off               # adaptive (default), always or off
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from cpr_engine import CoachConfig, StreamingDetector, simulate_session
from live_eval import evaluate_live, simulated_calibration

_PAUSES = tuple((s, 4.0) for s in (17.0, 38.0, 59.0, 80.0, 101.0))
_PERSON = dict(head_gain=0.9, head_coupling_jitter=0.04)

# name: (simulation settings, whether a spoken correction is expected)
CASES = {
    "on target (110, 5.5 cm)": (dict(rate_cpm=110, depth_m=0.055), False),
    "too slow (85)": (dict(rate_cpm=85, depth_m=0.055), True),
    "too fast (140)": (dict(rate_cpm=140, depth_m=0.055), True),
    "just slow (97)": (dict(rate_cpm=97, depth_m=0.055), True),
    "tiring: 115 down to 95": (dict(rate_cpm=115, rate_drift_cpm=-20, depth_m=0.055), False),
    "too shallow (4.2 cm)": (dict(rate_cpm=110, depth_m=0.042), True),
    "just shallow (4.8 cm)": (dict(rate_cpm=110, depth_m=0.048), True),
    "too deep (6.6 cm)": (dict(rate_cpm=110, depth_m=0.066), True),
    "slow and shallow": (dict(rate_cpm=88, depth_m=0.042), True),
    "pauses for breaths": (dict(rate_cpm=110, depth_m=0.055, pauses=_PAUSES), False),
    "uneven rhythm and depth": (
        dict(rate_cpm=110, depth_m=0.055, rate_jitter=0.10, depth_jitter=0.25), False),
    "noisy sensor, glances": (
        dict(rate_cpm=110, depth_m=0.055, noise_m=0.005, sway_m=0.03, n_glances=10), False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--show", help="print the spoken prompts for one case")
    parser.add_argument("--metronome", default="adaptive", choices=("adaptive", "always", "off"))
    args = parser.parse_args()

    calibration = simulated_calibration()
    config = CoachConfig(metronome_mode=args.metronome)

    if args.show:
        params, _ = CASES[args.show]
        r = evaluate_live(simulate_session(seed=0, **_PERSON, **params), calibration, config)
        print(f"Spoken prompts, {args.show} (metronome {args.metronome}):")
        for t, words in r.prompts:
            print(f"  {t:6.1f} s   {words}")
        if not r.prompts:
            print("  (none)")
        print("Metronome ticking:")
        for a, b in r.metronome_spans:
            print(f"  {a:6.1f} s to {b:.1f} s")
        if not r.metronome_spans:
            print("  (never)")
        return

    header = (
        f"{'case':<26}{'delay':>7}{'rate cue':>10}{'depth cue':>11}"
        f"{'false':>8}{'wrong':>7}{'need-':>7}{'first':>8}{'prompts':>9}{'tick':>7}"
    )
    sub = (
        f"{'':<26}{'ms':>7}{'right':>10}{'right':>11}"
        f"{'good':>8}{'way':>7}{'less':>7}{'fix s':>8}{'/min':>9}{'on':>7}"
    )
    print(f"Metronome: {args.metronome}")
    print(header)
    print(sub)
    print("-" * len(header))
    for name, (params, expects_fix) in CASES.items():
        rs = [
            evaluate_live(simulate_session(seed=seed, **_PERSON, **params), calibration, config)
            for seed in range(args.runs)
        ]

        def mean(key):
            return float(np.nanmean([getattr(r, key) for r in rs]))

        false_good = max(mean("rate_false_good"), mean("depth_false_good"))
        wrong_way = max(mean("rate_wrong_way"), mean("depth_wrong_way"))
        needless = max(mean("rate_needless"), mean("depth_needless"))
        first = f"{mean('first_correction_s'):.1f}" if expects_fix else "-"
        print(
            f"{name:<26}{mean('delay_ms'):>7.0f}{mean('rate_right'):>10.0%}"
            f"{mean('depth_right'):>11.0%}{false_good:>8.1%}{wrong_way:>7.1%}"
            f"{needless:>7.1%}{first:>8}{mean('prompts_per_min'):>9.1f}{mean('metronome_on'):>7.0%}"
        )

    print()
    print(f"Each case: {args.runs} two-minute sessions. Shares are of the time a true reading existed.")
    print("delay:      time from the bottom of a compression to the detector reporting it")
    print("cue right:  the cue matched what the truth clearly called for")
    print('false good: said "good" while the truth was out of range (worse of rate and depth)')
    print("wrong way:  said faster when too fast, harder when too deep, and so on")
    print("needless:   gave a correction while the truth was in range")
    print("first fix:  time from the first compression to the first spoken correction")
    print("tick on:    share of the session the metronome was sounding")
    print("Simulated people cannot hear the tick, so this shows when it would sound, not whether it helps.")

    # How long one sample takes to process.
    s = simulate_session(seed=0)
    det = StreamingDetector()
    start = time.perf_counter()
    for t, y in zip(s.t.tolist(), s.y.tolist()):
        det.update(t, y)
    per_sample = (time.perf_counter() - start) / s.t.size
    print()
    print(f"Processing one sample takes about {per_sample * 1e6:.0f} microseconds in Python.")


if __name__ == "__main__":
    main()
