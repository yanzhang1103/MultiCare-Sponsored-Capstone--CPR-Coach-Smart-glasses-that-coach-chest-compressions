"""Run the depth experiment on simulated people and print the scores.

    python run_depth_demo.py            # tables
    python run_depth_demo.py --plot     # also save depth_plot.png
"""

from __future__ import annotations

import argparse

import numpy as np

from cpr_engine import score_depth
from depth_experiment import (
    estimate_one_for_everyone,
    estimate_per_person,
    simulate_people,
)

SPREADS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="save depth_plot.png")
    parser.add_argument("--spread", type=float, default=0.08,
                        help="person-to-person variation in head gain (fraction)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # ---- Table 1: six people, both ways of calibrating -------------------
    people = simulate_people(6, gain_spread=args.spread, seed=args.seed)
    one = estimate_one_for_everyone(people)
    per = estimate_per_person(people)

    print(f"Six simulated people, head gain varying by {args.spread:.0%} between people")
    print()
    header = (
        f"{'':<8}{'':>6} | {'one calibration for everyone':^30} | {'calibrated per person':^30}"
    )
    sub = (
        f"{'person':<8}{'gain':>6} | {'err mm':>8}{'band':>8}{'false ok':>12}  |"
        f" {'err mm':>8}{'band':>8}{'false ok':>12}"
    )
    print(header)
    print(sub)
    print("-" * len(sub))
    for p, a, b in zip(people, one, per):
        print(f"{p.name:<8}{p.head_gain:>6.2f} | {_cells(a, p.test_depth)}  | {_cells(b, p.test_depth)}")
    truth = np.concatenate([p.test_depth for p in people])
    print("-" * len(sub))
    print(
        f"{'all':<8}{'':>6} | {_cells(np.concatenate(one), truth)}  |"
        f" {_cells(np.concatenate(per), truth)}"
    )
    print()
    print("err mm:   average depth error per compression, millimetres")
    print("band:     compressions put in the right band (too shallow / good 5 to 6 cm / too deep)")
    print("          proposed pass mark: 85%")
    print("false ok: compressions the glasses call good that the answer key does not")
    print("          guardrail: under 5%")

    # ---- Table 2: how much person-to-person variation is tolerable -------
    print()
    print("How the result changes with person-to-person variation (60 simulated people each)")
    print()
    head2 = (
        f"{'variation':>10} | {'one for everyone':^24} | {'per person':^24}"
    )
    sub2 = f"{'':>10} | {'err mm':>8}{'band':>8}{'false ok':>10} | {'err mm':>8}{'band':>8}{'false ok':>10}"
    print(head2)
    print(sub2)
    print("-" * len(sub2))
    rows = sweep_spread()
    for spread, sa, sb in rows:
        print(
            f"{spread:>10.0%} | {sa.mean_abs_error_mm:>8.1f}{sa.band_accuracy:>8.0%}"
            f"{sa.false_reassurance_fraction:>10.1%} | {sb.mean_abs_error_mm:>8.1f}"
            f"{sb.band_accuracy:>8.0%}{sb.false_reassurance_fraction:>10.1%}"
        )

    if args.plot:
        _plot(rows)


def _cells(est: np.ndarray, truth: np.ndarray) -> str:
    s = score_depth(est, truth)
    return f"{s.mean_abs_error_mm:>8.1f}{s.band_accuracy:>8.0%}{s.false_reassurance_fraction:>12.1%}"


def sweep_spread(spreads=SPREADS, n_people: int = 20, n_seeds: int = 3):
    rows = []
    for spread in spreads:
        one, per, truth = [], [], []
        for seed in range(n_seeds):
            people = simulate_people(n_people, gain_spread=spread, seed=seed)
            one += estimate_one_for_everyone(people)
            per += estimate_per_person(people)
            truth += [p.test_depth for p in people]
        t = np.concatenate(truth)
        rows.append(
            (spread, score_depth(np.concatenate(one), t), score_depth(np.concatenate(per), t))
        )
    return rows


def _plot(rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"

    x = [100 * r[0] for r in rows]
    one = [100 * r[1].band_accuracy for r in rows]
    per = [100 * r[2].band_accuracy for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=surface)
    ax.set_facecolor(surface)

    ax.axhline(85, color=ink2, lw=1, ls=(0, (4, 3)))
    ax.text(0.1, 85.6, "proposed pass mark: 85%", color=ink2, fontsize=9, va="bottom")

    ax.plot(x, per, color=blue, lw=2, marker="o", ms=7, mec=surface, mew=1.5,
            label="Calibrated per person")
    ax.plot(x, one, color=orange, lw=2, marker="s", ms=7, mec=surface, mew=1.5,
            label="One calibration for everyone")

    ax.text(x[-1] + 0.3, per[-1], "Per person", color=ink, fontsize=10, va="center")
    ax.text(x[-1] + 0.3, one[-1], "One for\neveryone", color=ink, fontsize=10, va="center")

    ax.set_xlim(-0.5, x[-1] + 2.6)
    ax.set_ylim(60, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}%" for v in x])
    ax.set_yticks(range(60, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(60, 101, 10)])
    ax.set_xlabel("Person-to-person variation in how far the head moves per cm of chest depth",
                  color=ink2, fontsize=10)
    ax.set_ylabel("Compressions put in the right depth band", color=ink2, fontsize=10)
    ax.set_title(
        "Simulated: one depth calibration for everyone holds only if people are alike",
        color=ink, fontsize=12, loc="left", pad=12,
    )
    ax.grid(axis="y", color=grid, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, length=0, labelsize=9)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(ink)

    fig.tight_layout()
    fig.savefig("depth_plot.png", dpi=130, facecolor=surface)
    print("\nsaved depth_plot.png")


if __name__ == "__main__":
    main()
