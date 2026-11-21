r"""Test the chest marker on generated frames, or find it in a real photo.

    python run_marker_demo.py                       # tables on generated frames
    python run_marker_demo.py --runs 80
    python run_marker_demo.py --image photo.jpg     # find the marker in a photo, save photo_marked.jpg
    python run_marker_demo.py --image photo.jpg --hand 812,640    # also check a hand at that pixel

For a photo: print chest_marker.pdf (python make_marker_sheet.py), lay it
down, and take a picture from above with any camera.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2

from cpr_engine.marker import MarkerLayout, check_hand, detect_marker, draw_overlay, target_px
from cpr_engine.marker_sim import Camera, View, evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=40, help="frames per row")
    parser.add_argument("--image", help="a photo to find the marker in")
    parser.add_argument("--hand", help="pixel x,y of a hand in the photo")
    parser.add_argument("--size-mm", type=float, default=80.0, help="printed marker size")
    parser.add_argument("--target-mm", default="0,120",
                        help="hand target from the marker centre: right,toward-feet in mm")
    args = parser.parse_args()

    tx, ty = (float(v) for v in args.target_mm.split(","))
    layout = MarkerLayout(marker_size_mm=args.size_mm, target_offset_mm=(tx, ty))

    if args.image:
        return photo(args.image, args.hand, layout)
    tables(args.runs, layout)
    return 0


def photo(path: str, hand: str | None, layout: MarkerLayout) -> int:
    image = cv2.imread(path)
    if image is None:
        print(f"ERROR  cannot read {path}")
        return 1
    sighting = detect_marker(image, layout)
    hand_px = tuple(float(v) for v in hand.split(",")) if hand else None
    out = Path(path).with_name(Path(path).stem + "_marked.jpg")
    cv2.imwrite(str(out), draw_overlay(image, sighting, layout, hand_px))
    if sighting is None:
        print("Marker not found.")
        print(f"saved {out}")
        return 1
    x, y = target_px(sighting, layout)
    print(f"Marker found: {sighting.width_px:.0f} pixels wide in a "
          f"{image.shape[1]} x {image.shape[0]} photo.")
    print(f"Hand target is at pixel ({x:.0f}, {y:.0f}).")
    if hand_px:
        c = check_hand(sighting, hand_px, layout)
        print(f"Hand is {c.distance_mm / 10:.1f} cm from the target: "
              + ("on target." if c.on_target else c.hint + "."))
    print(f"saved {out}")
    return 0


def tables(runs: int, layout: MarkerLayout) -> None:
    camera = Camera()
    base = View()
    print(f"Generated frames: {camera.width} x {camera.height} camera, {camera.hfov_deg:.0f} degree view, "
          f"{layout.marker_size_mm:.0f} mm marker, {runs} frames per row.")
    print("Each frame looks from a random direction. Target error is on the chest, in mm.")

    def show(title: str, label: str, rows) -> None:
        print()
        print(title)
        print(f"  {label:<22}{'marker px':>10}{'found':>8}{'target err mm':>15}{'worst 5%':>10}")
        for name, view in rows:
            r = evaluate(view, runs, layout, camera)
            err = f"{r.target_error_mm:.1f}" if r.found else "-"
            p95 = f"{r.target_error_p95_mm:.1f}" if r.found else "-"
            print(f"  {name:<22}{r.width_px:>10.0f}{r.found:>8.0%}{err:>15}{p95:>10}")

    show("Distance from camera to chest", "distance",
         [(f"{d} cm", replace(base, distance_mm=d * 10.0)) for d in (40, 50, 60, 70, 85, 100, 130)])
    show("Viewing angle (0 = straight down)", "angle",
         [(f"{a} degrees", replace(base, tilt_deg=float(a))) for a in (0, 20, 40, 55, 65, 75)])
    show("Motion blur (the head moves during a compression)", "blur",
         [(f"{b} pixels", replace(base, motion_blur_px=float(b))) for b in (0, 4, 8, 12, 16, 24)])
    show("Light and sensor noise", "condition", [
        ("normal", base),
        ("dim (30%)", replace(base, light=0.3)),
        ("dim and noisy", replace(base, light=0.3, noise=8.0)),
        ("very dim and noisy", replace(base, light=0.15, noise=8.0)),
    ])
    show("One corner of the marker covered", "covered", [
        (f"{int(c * 100)}% of its width", replace(base, covered=c)) for c in (0.0, 0.1, 0.2, 0.35)
    ])

    print()
    print(f"Hand position error when the hand point is {layout.hand_height_mm:.0f} mm above the chest")
    print(f"  {'angle':<22}{'flat method mm':>16}{'3D method mm':>14}")
    for a in (0, 10, 20, 30, 40):
        r = evaluate(replace(base, tilt_deg=float(a)), runs, layout, camera)
        print(f"  {str(a) + ' degrees':<22}{r.hand_flat_error_mm:>16.1f}{r.hand_3d_error_mm:>14.1f}")
    print("  flat: needs only the frame. 3D: also needs the camera's lens numbers.")


if __name__ == "__main__":
    sys.exit(main())
