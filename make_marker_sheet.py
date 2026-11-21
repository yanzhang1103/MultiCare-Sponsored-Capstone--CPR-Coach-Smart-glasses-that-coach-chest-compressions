r"""Make the printable chest marker sheet.

    python make_marker_sheet.py                     # chest_marker.pdf, 80 mm marker
    python make_marker_sheet.py --size-mm 100

Print it at 100% scale ("actual size", not "fit to page") on US Letter
paper, then check the ruler on the sheet with a real ruler.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cpr_engine.marker import (  # noqa: E402
    DICTIONARY_NAME,
    MARKER_ID,
    SHEET_BORDER_MM,
    make_marker_image,
)

PAGE_W_MM, PAGE_H_MM = 215.9, 279.4     # US Letter
RULER_MM = 100.0


def make_sheet(path: str, size_mm: float = 80.0) -> None:
    fig = plt.figure(figsize=(PAGE_W_MM / 25.4, PAGE_H_MM / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(0, PAGE_H_MM)
    ax.axis("off")

    cx, cy = PAGE_W_MM / 2.0, 165.0
    half = size_mm / 2.0
    outer = half + SHEET_BORDER_MM

    # The marker, at its exact size.
    ax.imshow(
        make_marker_image(600), cmap="gray", vmin=0, vmax=255, interpolation="nearest",
        extent=(cx - half, cx + half, cy - half, cy + half), zorder=2,
    )
    # Cut line around the white margin.
    ax.plot(
        [cx - outer, cx + outer, cx + outer, cx - outer, cx - outer],
        [cy - outer, cy - outer, cy + outer, cy + outer, cy - outer],
        color="0.6", lw=0.6, ls=(0, (4, 3)), zorder=1,
    )
    # Which way up: in the outer part of the margin, clear of the marker.
    ax.text(cx, cy + outer - 3.2, "▲  HEAD  ▲", ha="center", va="center",
            fontsize=9, color="0.25")
    ax.text(cx, cy - outer - 5, "cut along the dashed line", ha="center", va="center",
            fontsize=7, color="0.5")

    ax.text(cx, PAGE_H_MM - 22, "CPR coach chest marker", ha="center", fontsize=16)
    ax.text(cx, PAGE_H_MM - 30,
            f"{DICTIONARY_NAME}, number {MARKER_ID}, {size_mm:.0f} mm square",
            ha="center", fontsize=9, color="0.3")

    # Ruler to check the print scale.
    ry = 62.0
    rx = cx - RULER_MM / 2.0
    ax.plot([rx, rx + RULER_MM], [ry, ry], color="black", lw=1.0)
    for i in range(int(RULER_MM / 10) + 1):
        ax.plot([rx + 10 * i] * 2, [ry - 2.5, ry + 2.5], color="black", lw=1.0)
    ax.text(cx, ry - 8,
            f"This line must measure {RULER_MM:.0f} mm. If it does not, print again at 100% scale "
            "(turn off \"fit to page\").",
            ha="center", fontsize=8)

    steps = [
        "1. Cut out the square and fix it flat on the manikin's chest, HEAD arrows toward the head.",
        "2. Keep it where the hands will not cover it: above the hand position, toward the neck.",
        "3. Measure from the centre of the marker to the centre of the correct hand position.",
        "4. Give that distance to the code as the target offset (for example --target-mm 0,120).",
    ]
    for i, line in enumerate(steps):
        ax.text(28, 42 - 6 * i, line, fontsize=8, color="0.2")

    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size-mm", type=float, default=80.0)
    parser.add_argument("-o", "--output", default="chest_marker.pdf")
    args = parser.parse_args()
    make_sheet(args.output, args.size_mm)
    print(f"saved {args.output}")
    print("Print at 100% scale, then check the ruler on the sheet.")


if __name__ == "__main__":
    main()
