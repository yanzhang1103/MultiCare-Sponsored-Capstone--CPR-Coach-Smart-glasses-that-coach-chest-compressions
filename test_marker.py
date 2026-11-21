import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpr_engine.marker import (  # noqa: E402
    MarkerLayout,
    check_hand,
    detect_marker,
    draw_overlay,
    hand_position_mm,
    make_marker_image,
    target_px,
)
from cpr_engine.marker_sim import Camera, View, evaluate, render  # noqa: E402
from make_marker_sheet import make_sheet  # noqa: E402

LAYOUT = MarkerLayout()
CAMERA = Camera()


def _target_error_mm(frame, sighting, layout=LAYOUT):
    t_px = np.array([[target_px(sighting, layout)]], dtype=np.float64)
    t_mm = cv2.perspectiveTransform(t_px, np.linalg.inv(frame.mm_to_px)).reshape(2)
    return float(np.linalg.norm(t_mm - np.array(layout.target_offset_mm)))


def test_finds_marker_and_target():
    frame = render(View(), LAYOUT, CAMERA, seed=1)
    s = detect_marker(frame.image, LAYOUT)
    assert s is not None
    assert np.abs(s.corners_px - frame.corners_px).max() < 1.0
    assert _target_error_mm(frame, s) < 2.0


def test_holds_up_across_views():
    for view in (View(distance_mm=450), View(distance_mm=900), View(tilt_deg=55),
                 View(light=0.3, noise=8.0), View(motion_blur_px=8)):
        r = evaluate(view, 12, LAYOUT, CAMERA)
        assert r.found == 1.0
        assert r.target_error_p95_mm < 4.0


def test_any_rotation_keeps_the_head_edge():
    # The target must stay on the "feet" side of the marker however the image is turned.
    for roll in (0, 90, 180, 270, 33):
        frame = render(View(roll_deg=roll, tilt_deg=10), LAYOUT, CAMERA, seed=2)
        s = detect_marker(frame.image, LAYOUT)
        assert _target_error_mm(frame, s) < 2.0


def test_no_marker_means_none():
    blank = np.full((720, 1280), 120, np.uint8)
    assert detect_marker(blank, LAYOUT) is None


def test_other_marker_numbers_are_ignored():
    canvas = np.full((800, 800), 255, np.uint8)
    canvas[200:600, 200:600] = make_marker_image(400, marker_id=3)
    assert detect_marker(canvas, LAYOUT) is None
    canvas[200:600, 200:600] = make_marker_image(400)
    assert detect_marker(canvas, LAYOUT) is not None


def test_heavy_blur_is_a_miss_not_a_wrong_answer():
    r = evaluate(View(motion_blur_px=40), 8, LAYOUT, CAMERA)
    assert r.found == 0.0


def test_covered_marker_is_a_miss():
    r = evaluate(View(covered=0.4), 8, LAYOUT, CAMERA)
    assert r.found == 0.0


def test_mm_px_round_trip():
    s = detect_marker(render(View(tilt_deg=35), LAYOUT, CAMERA, seed=3).image, LAYOUT)
    pts = np.array([[0.0, 0.0], [25.0, -10.0], [-40.0, 120.0]])
    assert np.allclose(s.to_mm(s.to_px(pts)), pts, atol=1e-6)


def test_3d_method_removes_the_slant_error():
    frame = render(View(tilt_deg=30, azimuth_deg=40, roll_deg=25), LAYOUT, CAMERA, seed=4)
    s = detect_marker(frame.image, LAYOUT)
    tx, ty = LAYOUT.target_offset_mm
    hand_px = frame.project([[tx, ty, -LAYOUT.hand_height_mm]])[0]

    flat = np.array(hand_position_mm(s, hand_px, LAYOUT))
    full = np.array(hand_position_mm(s, hand_px, LAYOUT, CAMERA.matrix))
    flat_err = np.linalg.norm(flat - [tx, ty])
    assert np.linalg.norm(full - [tx, ty]) < 2.0
    # Flat is off by about hand height x tan(angle): 40 mm x tan(30 degrees) = 23 mm.
    assert 15.0 < flat_err < 32.0


@pytest.mark.parametrize(
    "offset, words",
    [((0, 60), "toward the head"), ((0, -60), "toward the feet"),
     ((60, 0), "toward the marker's left"), ((-60, 0), "toward the marker's right")],
)
def test_hint_points_back_to_the_target(offset, words):
    layout = replace(LAYOUT, hand_height_mm=0.0)
    frame = render(View(tilt_deg=0), layout, CAMERA, seed=5)
    s = detect_marker(frame.image, layout)
    tx, ty = layout.target_offset_mm
    hand_px = frame.project([[tx + offset[0], ty + offset[1], 0.0]])[0]
    c = check_hand(s, hand_px, layout)
    assert not c.on_target
    assert c.distance_mm == pytest.approx(60.0, abs=2.0)
    assert words in c.hint and "6 cm" in c.hint


def test_hand_on_target():
    layout = replace(LAYOUT, hand_height_mm=0.0)
    frame = render(View(tilt_deg=0), layout, CAMERA, seed=6)
    s = detect_marker(frame.image, layout)
    tx, ty = layout.target_offset_mm
    c = check_hand(s, frame.project([[tx + 10.0, ty - 15.0, 0.0]])[0], layout)
    assert c.on_target and c.hint == ""


def test_overlay():
    frame = render(View(), LAYOUT, CAMERA, seed=7)
    s = detect_marker(frame.image, LAYOUT)
    out = draw_overlay(frame.image, s, LAYOUT, hand_px=(900, 500))
    assert out.shape == (CAMERA.height, CAMERA.width, 3)
    assert draw_overlay(frame.image, None, LAYOUT).shape == out.shape


def test_sheet_is_written(tmp_path):
    path = tmp_path / "marker.pdf"
    make_sheet(str(path), 80.0)
    assert path.stat().st_size > 2000
    assert path.read_bytes()[:4] == b"%PDF"
