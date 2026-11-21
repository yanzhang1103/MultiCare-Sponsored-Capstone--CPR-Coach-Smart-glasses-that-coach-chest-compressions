"""The chest marker: find it in a camera frame and locate the hand target.

A printed marker is fixed to the manikin's chest, a known distance from
where the hands should go. Finding the marker in a frame therefore
tells us where the target spot is in that frame, and how far a hand is
from it in millimetres.

The marker is an ArUco marker from the "4x4_50" set. Both OpenCV (used
here, and for frames from the XREAL Eye camera) and Android XR's
built-in marker tracking (used on the Aura) read that set, so one
printed marker serves both glasses.

Marker coordinates, in millimetres, measured on the chest:
    origin  the centre of the marker
    +x      toward the marker's right edge
    +y      toward the marker's bottom edge (the feet, when the printed
            "HEAD" arrow points to the manikin's head)

Two ways to place a hand on that map:
    flat    treat the hand as lying in the plane of the marker. Needs
            nothing but the frame. Off by (hand height above the chest)
            x tan(viewing angle), which is centimetres at a slant.
    3D      with the camera's lens numbers (camera matrix), allow for
            the hand sitting above the chest. Removes that error.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_50
DICTIONARY_NAME = "ArUco 4x4_50"
MARKER_ID = 7
SHEET_BORDER_MM = 20.0      # white margin around the marker on the printed sheet


@dataclass
class MarkerLayout:
    """Where things are on the chest. Measure these on the real manikin."""

    marker_size_mm: float = 80.0                          # black square, edge to edge
    target_offset_mm: tuple[float, float] = (0.0, 120.0)  # hand target, from the marker centre
    tolerance_mm: float = 30.0                            # a hand this close to the target is "on it"
    hand_height_mm: float = 40.0                          # how far the tracked hand point sits above the chest


@dataclass
class MarkerSighting:
    """One detection of the marker in one frame."""

    corners_px: np.ndarray        # 4 x 2: top-left, top-right, bottom-right, bottom-left
    width_px: float               # average side length in the image
    mm_to_px: np.ndarray          # 3 x 3 map from marker millimetres to image pixels
    px_to_mm: np.ndarray          # and back

    def to_px(self, points_mm) -> np.ndarray:
        """Marker-plane points (mm) to image pixels."""
        return _apply(self.mm_to_px, points_mm)

    def to_mm(self, points_px) -> np.ndarray:
        """Image pixels to marker-plane millimetres (assumes the point lies on the chest)."""
        return _apply(self.px_to_mm, points_px)


@dataclass
class HandCheck:
    position_mm: tuple[float, float]   # where the hand is, in marker coordinates
    move_mm: tuple[float, float]       # how far to move it to reach the target
    distance_mm: float
    on_target: bool
    hint: str                          # "" when on target, else e.g. "move 4 cm toward the feet"


def _apply(h: np.ndarray, points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, h).reshape(-1, 2)


def marker_corners_mm(size_mm: float) -> np.ndarray:
    """The marker's four corners in marker coordinates, in OpenCV's corner order."""
    h = size_mm / 2.0
    return np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)


def make_marker_image(size_px: int = 600, marker_id: int = MARKER_ID) -> np.ndarray:
    """The marker as a black and white image, for printing."""
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)


_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # The marker can be small in a head-mounted view; allow small candidates.
        params.minMarkerPerimeterRate = 0.01
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 10
        _DETECTOR = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(DICTIONARY), params
        )
    return _DETECTOR


def detect_marker(
    image: np.ndarray,
    layout: MarkerLayout | None = None,
    marker_id: int = MARKER_ID,
) -> MarkerSighting | None:
    """Find the chest marker in a frame. Returns None when it is not visible.

    image: a colour (BGR) or grey frame.
    """
    layout = layout or MarkerLayout()
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(grey)
    if ids is None:
        return None
    for c, found_id in zip(corners, ids.flatten()):
        if int(found_id) != marker_id:
            continue
        px = c.reshape(4, 2).astype(np.float64)
        mm = marker_corners_mm(layout.marker_size_mm)
        mm_to_px = cv2.getPerspectiveTransform(mm.astype(np.float32), px.astype(np.float32))
        sides = [np.linalg.norm(px[i] - px[(i + 1) % 4]) for i in range(4)]
        return MarkerSighting(
            corners_px=px,
            width_px=float(np.mean(sides)),
            mm_to_px=mm_to_px,
            px_to_mm=np.linalg.inv(mm_to_px),
        )
    return None


def target_px(sighting: MarkerSighting, layout: MarkerLayout | None = None) -> tuple[float, float]:
    """Where the hand target is in the image."""
    layout = layout or MarkerLayout()
    x, y = sighting.to_px([layout.target_offset_mm])[0]
    return float(x), float(y)


def hand_position_mm(
    sighting: MarkerSighting,
    hand_px,
    layout: MarkerLayout | None = None,
    camera_matrix: np.ndarray | None = None,
) -> tuple[float, float]:
    """Where a hand seen at hand_px is, in marker coordinates.

    Without camera_matrix: the flat method (hand assumed to lie on the chest).
    With camera_matrix:    the 3D method, which allows for layout.hand_height_mm.
    """
    layout = layout or MarkerLayout()
    if camera_matrix is None:
        x, y = sighting.to_mm([hand_px])[0]
        return float(x), float(y)

    # Work out where the camera is relative to the marker, then follow the
    # line of sight through the hand's pixel down to the hand's height.
    half = layout.marker_size_mm / 2.0
    object_pts = np.array(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=np.float64
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_pts, sighting.corners_px, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        x, y = sighting.to_mm([hand_px])[0]
        return float(x), float(y)
    rot, _ = cv2.Rodrigues(rvec)
    cam_pos = (-rot.T @ tvec).ravel()                      # camera position, marker coordinates
    ray_cam = np.linalg.inv(camera_matrix) @ np.array([hand_px[0], hand_px[1], 1.0])
    ray = rot.T @ ray_cam                                   # line of sight, marker coordinates
    # +z points into the chest, so a hand above the chest is at z = -height.
    z_hand = -layout.hand_height_mm
    if abs(ray[2]) < 1e-9:
        x, y = sighting.to_mm([hand_px])[0]
        return float(x), float(y)
    s = (z_hand - cam_pos[2]) / ray[2]
    hit = cam_pos + s * ray
    return float(hit[0]), float(hit[1])


def check_hand(
    sighting: MarkerSighting,
    hand_px,
    layout: MarkerLayout | None = None,
    camera_matrix: np.ndarray | None = None,
) -> HandCheck:
    """Is a hand seen at hand_px on the target, and if not, which way should it move?"""
    layout = layout or MarkerLayout()
    x, y = hand_position_mm(sighting, hand_px, layout, camera_matrix)
    dx = layout.target_offset_mm[0] - x
    dy = layout.target_offset_mm[1] - y
    distance = float(np.hypot(dx, dy))
    on_target = distance <= layout.tolerance_mm
    return HandCheck(
        position_mm=(x, y),
        move_mm=(dx, dy),
        distance_mm=distance,
        on_target=on_target,
        hint="" if on_target else _hint(dx, dy),
    )


def _hint(dx: float, dy: float) -> str:
    """The larger part of the correction, in words."""
    if abs(dy) >= abs(dx):
        direction = "toward the feet" if dy > 0 else "toward the head"
        amount = abs(dy)
    else:
        direction = "toward the marker's right" if dx > 0 else "toward the marker's left"
        amount = abs(dx)
    return f"move {amount / 10.0:.0f} cm {direction}"


def draw_overlay(
    image: np.ndarray,
    sighting: MarkerSighting | None,
    layout: MarkerLayout | None = None,
    hand_px=None,
    camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """A copy of the frame with the marker outline, the target zone and (if given) the hand."""
    layout = layout or MarkerLayout()
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = max(1, int(round(max(out.shape[:2]) / 900)))
    if sighting is None:
        cv2.putText(out, "marker not found", (20 * scale, 40 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, (0, 0, 255), 2 * scale)
        return out

    cv2.polylines(out, [sighting.corners_px.astype(np.int32)], True, (0, 200, 0), 2 * scale)
    top = sighting.corners_px[:2].mean(axis=0)
    cv2.circle(out, tuple(int(v) for v in top), 5 * scale, (0, 200, 0), -1)   # the "head" edge

    # The target zone is a circle on the chest, so it shows as an ellipse.
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    cx, cy = layout.target_offset_mm
    ring = np.column_stack(
        (cx + layout.tolerance_mm * np.cos(angles), cy + layout.tolerance_mm * np.sin(angles))
    )
    ring_px = sighting.to_px(ring).astype(np.int32)
    colour = (255, 160, 0)
    if hand_px is not None:
        result = check_hand(sighting, hand_px, layout, camera_matrix)
        colour = (0, 200, 0) if result.on_target else (0, 0, 255)
        cv2.circle(out, (int(hand_px[0]), int(hand_px[1])), 7 * scale, colour, -1)
        label = "on target" if result.on_target else result.hint
        cv2.putText(out, label, (20 * scale, 40 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, colour, 2 * scale)
    cv2.polylines(out, [ring_px], True, colour, 2 * scale)
    tx, ty = target_px(sighting, layout)
    cv2.drawMarker(out, (int(tx), int(ty)), colour, cv2.MARKER_CROSS, 16 * scale, 2 * scale)
    return out
