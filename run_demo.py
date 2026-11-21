"""Run the rate and pause detector on simulated sessions and print the scores.

    python run_demo.py            # table of all scenarios
    python run_demo.py --plot     # also save demo_plot.png for one scenario
"""

from __future__ import annotations

import argparse

import numpy as np

from cpr_engine import (
    detect_compressions,
    find_pauses,
    match_events,
    rate_series,
    score_pauses,
    score_rate,
    simulate_session,
)
from scenarios import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="save demo_plot.png")
    parser.add_argument("--scenario", default="everything at once")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    header = (
        f"{'scenario':<26}{'true':>6}{'found':>7}{'missed':>8}{'extra':>7}"
        f"{'within 5':>10}{'avg err':>9}{'pauses':>8}{'result':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, params in SCENARIOS.items():
        s = simulate_session(seed=args.seed, **params)
        est = detect_compressions(s.t, s.y)
        ev = match_events(est, s.true_times)
        rs = score_rate(est, s.true_times, s.duration_s)
        ps = score_pauses(est, s.true_times)
        ok = rs.passed and rs.guardrail_passed
        print(
            f"{name:<26}{ev.n_true:>6}{ev.n_detected:>7}{ev.n_missed:>8}{ev.n_extra:>7}"
            f"{f'{rs.n_within}/{rs.n_segments}':>10}{rs.mean_abs_error_cpm:>9.1f}"
            f"{f'{ps.n_found}/{ps.n_true}':>8}{'PASS' if ok else 'FAIL':>8}"
        )

    print()
    print("true/found: compressions in the answer key / counted by the detector")
    print("within 5:   30-second segments where the rate is within 5 per minute")
    print("avg err:    average rate error, compressions per minute")
    print("pauses:     pauses found / pauses in the answer key")
    print("result:     PASS needs 90% of segments within 5 and the guardrail under 5%")

    if args.plot:
        _plot(args.scenario, args.seed)


def _plot(name: str, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = simulate_session(seed=seed, **SCENARIOS[name])
    est = detect_compressions(s.t, s.y)
    t_eval = np.arange(5.0, s.duration_s, 0.5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    ax1.plot(s.t, s.y * 100, color="#555", lw=0.8, label="head height")
    ax1.plot(est, np.interp(est, s.t, s.y) * 100, "v", color="#d1495b", ms=4,
             label="detected compression")
    for a, b in find_pauses(est):
        ax1.axvspan(a, b, color="#edae49", alpha=0.25)
    ax1.set_ylabel("head height (cm)")
    ax1.set_title(f"Simulated session: {name}  (shaded = detected pause)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.axhspan(100, 120, color="#66a182", alpha=0.2, label="target 100 to 120")
    ax2.plot(t_eval, rate_series(s.true_times, t_eval), color="#00798c", lw=2,
             label="true rate")
    ax2.plot(t_eval, rate_series(est, t_eval), color="#d1495b", lw=1.2, ls="--",
             label="detected rate")
    ax2.set_ylabel("rate (per minute, 5 s window)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig("demo_plot.png", dpi=130)
    print("\nsaved demo_plot.png")


if __name__ == "__main__":
    main()
