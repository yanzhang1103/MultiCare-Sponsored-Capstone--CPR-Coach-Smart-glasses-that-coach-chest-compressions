"""Generated camera frames of the chest marker, with a known answer.

Stands in for real frames from the glasses until the camera arrives.
A simple pinhole camera looks at the printed sheet from a chosen
distance and angle; blur, dim light, noise and a covered corner can be
added. It proves the code works and shows where detection gives out.
Real frames (lens distortion, glare, fabric, a curved chest) will be
harder in ways this does not model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .marker import SHEET_BORDER_MM, MarkerLayout, make_marker_image, marker_corners_mm

SHEET_PX_PER_MM = 8.0


@dataclass
class Camera:
    width: int = 1920
    height: int = 1080
    hfov_deg: float = 80.0      # horizontal field of view

    @property
    def matrix(self) -> np.ndarray:
        f = (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)
        return np.array([[f, 0, self.width / 2.0], [0, f, self.height / 2.0], [0, 0, 1.0]])


@dataclass
class View:
    distance_mm: float = 600.0      # camera to the chest
    tilt_deg: float = 20.0          # 0 = looking straight down at the chest
    azimuth_deg: float = 0.0        # which side the camera leans toward
    roll_deg: float = 0.0           # rotation of the image about the line of sight
    motion_blur_px: float = 0.0
    blur_sigma_px: float = 0.8
    light: float = 1.0              # 1 = normal, 0.3 = dim
    noise: float = 3.0              # grey levels of sensor noise
    covered: float = 0.0            # share of the marker's width hidden at one corner


@dataclass
class Frame:
    image: np.ndarray               # grey frame
    corners_px: np.ndarray          # true marker corners
    mm_to_px: np.ndarray            # true map from marker mm to pixels
    rotation: np.ndarray            # camera rotation (marker -> camera)
    camera_pos_mm: np.ndarray       # camera position in marker coordinates
    camera: Camera

    def project(self, points_mm_3d) -> np.ndarray:
        """3D points in marker coordinates (mm) to pixels."""
        p = np.asarray(points_mm_3d, dtype=float).reshape(-1, 3)
        cam = (self.rotation @ (p - self.camera_pos_mm).T).T
        px = (self.camera.matrix @ cam.T).T
        return px[:, :2] / px[:, 2:3]


def render(view: View, layout: MarkerLayout | None = None, camera: Camera | None = None,
           seed: int = 0) -> Frame:
    layout = layout or MarkerLayout()
    camera = camera or Camera()
    rng = np.random.default_rng(seed)

    # Camera pose. Marker axes: x right, y down, z into the chest.
    look = np.array([layout.target_offset_mm[0] / 2.0, layout.target_offset_mm[1] / 2.0, 0.0])
    tilt, az = np.radians(view.tilt_deg), np.radians(view.azimuth_deg)
    out_of_chest = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
    cam_pos = look + view.distance_mm * out_of_chest
    z = -out_of_chest
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    roll = np.radians(view.roll_deg)
    x, y = np.cos(roll) * x + np.sin(roll) * y, -np.sin(roll) * x + np.cos(roll) * y
    rotation = np.vstack((x, y, z))

    frame = Frame(
        image=np.zeros((camera.height, camera.width), np.uint8),
        corners_px=np.zeros((4, 2)),
        mm_to_px=np.eye(3),
        rotation=rotation,
        camera_pos_mm=cam_pos,
        camera=camera,
    )
    corners_mm = marker_corners_mm(layout.marker_size_mm)
    frame.corners_px = frame.project(np.column_stack((corners_mm, np.zeros(4))))
    frame.mm_to_px = cv2.getPerspectiveTransform(
        corners_mm.astype(np.float32), frame.corners_px.astype(np.float32)
    )

    # The printed sheet: the marker on white paper.
    size_px = int(round(layout.marker_size_mm * SHEET_PX_PER_MM))
    border_px = int(round(SHEET_BORDER_MM * SHEET_PX_PER_MM))
    sheet = np.full((size_px + 2 * border_px,) * 2, 255, np.uint8)
    sheet[border_px:border_px + size_px, border_px:border_px + size_px] = make_marker_image(size_px)
    if view.covered > 0:
        # Something dark over one corner, like the edge of a hand or sleeve.
        n = int(round(view.covered * size_px))
        sheet[border_px:border_px + n, border_px:border_px + n] = 70

    half = layout.marker_size_mm / 2.0 + SHEET_BORDER_MM
    sheet_mm = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    sheet_px = frame.project(np.column_stack((sheet_mm, np.zeros(4))))
    n_sheet = sheet.shape[0]
    src = np.array([[0, 0], [n_sheet, 0], [n_sheet, n_sheet], [0, n_sheet]], np.float32)
    warp = cv2.getPerspectiveTransform(src, sheet_px.astype(np.float32))
    size = (camera.width, camera.height)
    warped = cv2.warpPerspective(sheet, warp, size, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full_like(sheet, 255), warp, size, flags=cv2.INTER_AREA)

    # Background: uneven mid-grey, like clothing.
    coarse = rng.normal(115, 22, (camera.height // 60 + 1, camera.width // 60 + 1)).astype(np.float32)
    background = cv2.resize(coarse, size, interpolation=cv2.INTER_CUBIC)
    alpha = mask.astype(np.float32) / 255.0
    img = alpha * warped.astype(np.float32) + (1.0 - alpha) * background

    img *= view.light
    if view.motion_blur_px >= 2:
        n = int(round(view.motion_blur_px))
        kernel = np.zeros((n, n), np.float32)
        kernel[n // 2, :] = 1.0 / n
        angle = float(rng.uniform(0, 180))
        rot = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot, (n, n))
        kernel /= max(kernel.sum(), 1e-6)
        img = cv2.filter2D(img, -1, kernel)
    if view.blur_sigma_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), view.blur_sigma_px)
    if view.noise > 0:
        img += rng.normal(0, view.noise, img.shape).astype(np.float32)
    frame.image = np.clip(img, 0, 255).astype(np.uint8)
    return frame


@dataclass
class MarkerResult:
    n: int
    found: float                # share of frames where the marker was found
    width_px: float             # typical marker width in the image
    target_error_mm: float      # typical error in the target's position (found frames)
    target_error_p95_mm: float
    hand_flat_error_mm: float   # typical error in a hand's position, flat method
    hand_3d_error_mm: float     # ... and with the camera matrix


def evaluate(view: View, n: int = 40, layout: MarkerLayout | None = None,
             camera: Camera | None = None, seed: int = 0) -> MarkerResult:
    """Render n frames like `view` from random directions and score the detector."""
    from dataclasses import replace

    from .marker import detect_marker, hand_position_mm, target_px

    layout = layout or MarkerLayout()
    camera = camera or Camera()
    rng = np.random.default_rng(seed)
    target = np.array(layout.target_offset_mm, dtype=float)
    hand_3d = [target[0], target[1], -layout.hand_height_mm]   # a hand exactly on target

    found = 0
    widths, target_err, flat_err, full_err = [], [], [], []
    for i in range(n):
        v = replace(view, azimuth_deg=float(rng.uniform(0, 360)), roll_deg=float(rng.uniform(0, 360)))
        frame = render(v, layout, camera, seed=seed * 10007 + i)
        widths.append(float(np.mean(
            [np.linalg.norm(frame.corners_px[k] - frame.corners_px[(k + 1) % 4]) for k in range(4)]
        )))
        sighting = detect_marker(frame.image, layout)
        if sighting is None:
            continue
        found += 1
        true_px_to_mm = np.linalg.inv(frame.mm_to_px)
        t_px = np.array([[target_px(sighting, layout)]], dtype=np.float64)
        t_mm = cv2.perspectiveTransform(t_px, true_px_to_mm).reshape(2)
        target_err.append(float(np.linalg.norm(t_mm - target)))

        hand_px = frame.project([hand_3d])[0]
        flat = np.array(hand_position_mm(sighting, hand_px, layout))
        full = np.array(hand_position_mm(sighting, hand_px, layout, camera.matrix))
        flat_err.append(float(np.linalg.norm(flat - target)))
        full_err.append(float(np.linalg.norm(full - target)))

    def med(values):
        return float(np.median(values)) if values else float("nan")

    return MarkerResult(
        n=n,
        found=found / n,
        width_px=med(widths),
        target_error_mm=med(target_err),
        target_error_p95_mm=float(np.percentile(target_err, 95)) if target_err else float("nan"),
        hand_flat_error_mm=med(flat_err),
        hand_3d_error_mm=med(full_err),
    )
