"""Read and write recorded sessions.

One session is one folder. The layout and every column are described
in SESSION_FORMAT.md; this file is the code version of that document.

    <session folder>/
        session.json     who, what, when
        head_pose.csv    from the glasses capture app
        manikin.csv      answer key from the manikin (absent for pillow sessions)

The loader checks the files as it reads them, because a wrong unit or
a wrong axis would otherwise pass silently and spoil every score.
Problems that make the data unusable raise SessionError. Things that
look odd but may be fine are collected in Session.warnings.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .depth import Calibration

FORMAT_VERSION = 1
META_FILE = "session.json"
POSE_FILE = "head_pose.csv"
MANIKIN_FILE = "manikin.csv"


class SessionError(ValueError):
    """The session's files cannot be used as they are."""


@dataclass
class Session:
    path: Path
    meta: dict
    t: np.ndarray                          # head pose sample times, seconds
    height: np.ndarray                     # head height, metres
    manikin_t: np.ndarray | None = None    # manikin's time of each compression, its own clock
    manikin_depth_m: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        return str(self.meta.get("session_id") or self.path.name)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def has_answer_key(self) -> bool:
        return self.manikin_t is not None


def load_session(folder) -> Session:
    folder = Path(folder)
    if not folder.is_dir():
        raise SessionError(f"{folder} is not a folder")

    warnings: list[str] = []

    meta_path = folder / META_FILE
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            raise SessionError(f"{META_FILE} is not valid JSON: {e}") from e
    else:
        meta = {}
        warnings.append(f"no {META_FILE}; using defaults (up axis y)")

    # ---- head pose -------------------------------------------------------
    pose = _read_csv(folder / POSE_FILE)
    axis = str(meta.get("up_axis", "y")).lower()
    if axis not in ("x", "y", "z"):
        raise SessionError(f'{META_FILE}: up_axis must be "x", "y" or "z", not "{axis}"')
    t = _column(pose, "t_s", POSE_FILE)
    height = _column(pose, f"{axis}_m", POSE_FILE)

    if t.size < 50:
        raise SessionError(f"{POSE_FILE} has only {t.size} rows")
    steps = np.diff(t)
    if np.any(steps <= 0):
        n_bad = int(np.count_nonzero(steps <= 0))
        if n_bad > 0.01 * t.size:
            raise SessionError(
                f"{POSE_FILE}: t_s goes backwards or repeats in {n_bad} rows; "
                "it must come from a clock that only moves forward"
            )
        order = np.argsort(t, kind="stable")
        t, height = t[order], height[order]
        keep = np.concatenate(([True], np.diff(t) > 0))
        t, height = t[keep], height[keep]
        warnings.append(f"{POSE_FILE}: {n_bad} rows were out of time order and were fixed")
        steps = np.diff(t)

    typical_step = float(np.median(steps))
    if typical_step > 0.5:
        raise SessionError(
            f"{POSE_FILE}: rows are {typical_step:.0f} apart in t_s. "
            "t_s must be in seconds; this looks like milliseconds"
        )
    rate_hz = 1.0 / typical_step
    if rate_hz < 20.0:
        warnings.append(f"{POSE_FILE}: only {rate_hz:.0f} samples a second; 30 or more is needed")
    if steps.max() > 0.25:
        where = float(t[int(np.argmax(steps))])
        warnings.append(
            f"{POSE_FILE}: a gap of {steps.max():.2f} s with no samples at t = {where:.1f} s"
        )
    if (t[-1] - t[0]) < 20.0:
        warnings.append(f"session is only {t[-1] - t[0]:.0f} s long")

    lo, hi = np.percentile(height, [1, 99])
    if hi - lo > 1.0:
        raise SessionError(
            f"{POSE_FILE}: {axis}_m ranges over {hi - lo:.1f}; head height must be in "
            "metres and should move by centimetres. Check the units"
        )
    if hi - lo < 0.005:
        warnings.append(
            f"{POSE_FILE}: {axis}_m barely changes ({(hi - lo) * 1000:.1f} mm). "
            "Is up_axis right, and were there compressions?"
        )

    # ---- manikin answer key (optional) ------------------------------------
    manikin_t = manikin_depth = None
    manikin_path = folder / MANIKIN_FILE
    if manikin_path.exists():
        man = _read_csv(manikin_path)
        manikin_t = _column(man, "t_s", MANIKIN_FILE)
        order = np.argsort(manikin_t, kind="stable")
        manikin_t = manikin_t[order]
        if "depth_m" in man:
            manikin_depth = man["depth_m"][order]
            if np.any(np.isnan(manikin_depth)):
                raise SessionError(f"{MANIKIN_FILE}: depth_m has empty or non-numeric cells")
            typical = float(np.median(manikin_depth))
            if typical > 0.15:
                raise SessionError(
                    f"{MANIKIN_FILE}: typical depth_m is {typical:g}. Depth must be in "
                    "metres (5 cm is 0.05); this looks like centimetres or millimetres"
                )
            if typical < 0.005:
                raise SessionError(
                    f"{MANIKIN_FILE}: typical depth_m is {typical:g} m, under 5 mm. Check the units"
                )
        else:
            warnings.append(f"{MANIKIN_FILE} has no depth_m column; depth will not be scored")
        if manikin_t.size < 10:
            warnings.append(f"{MANIKIN_FILE} lists only {manikin_t.size} compressions")

    return Session(
        path=folder,
        meta=meta,
        t=t,
        height=height,
        manikin_t=manikin_t,
        manikin_depth_m=manikin_depth,
        warnings=warnings,
    )


def save_session(
    folder,
    meta: dict,
    t,
    position_m,
    rotation_xyzw=None,
    manikin_t=None,
    manikin_depth_m=None,
) -> Path:
    """Write a session folder in the standard layout.

    position_m:    N x 3 array of x, y, z in metres.
    rotation_xyzw: N x 4 quaternions, or None to write no rotation.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    meta = {"format_version": FORMAT_VERSION, **meta}
    (folder / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    t = np.asarray(t, dtype=float)
    pos = np.asarray(position_m, dtype=float)
    header = ["t_s", "x_m", "y_m", "z_m"]
    cols = [t, pos[:, 0], pos[:, 1], pos[:, 2]]
    if rotation_xyzw is not None:
        rot = np.asarray(rotation_xyzw, dtype=float)
        header += ["qx", "qy", "qz", "qw"]
        cols += [rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]]
    _write_csv(folder / POSE_FILE, header, cols)

    if manikin_t is not None:
        header = ["t_s"]
        cols = [np.asarray(manikin_t, dtype=float)]
        if manikin_depth_m is not None:
            header.append("depth_m")
            cols.append(np.asarray(manikin_depth_m, dtype=float))
        _write_csv(folder / MANIKIN_FILE, header, cols)
    return folder


def save_calibration(path, cal: Calibration, info: dict | None = None) -> None:
    data = {
        "format_version": FORMAT_VERSION,
        "gain": cal.gain,
        "offset_m": cal.offset_m,
        **(info or {}),
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_calibration(path) -> Calibration:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return Calibration(gain=float(data["gain"]), offset_m=float(data["offset_m"]))
    except (OSError, KeyError, ValueError) as e:
        raise SessionError(f"cannot read calibration file {path}: {e}") from e


# ---- helpers ---------------------------------------------------------------


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise SessionError(f"missing file: {path.name}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip().lower() for h in next(reader)]
        except StopIteration:
            raise SessionError(f"{path.name} is empty") from None
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise SessionError(f"{path.name} has a header but no data")

    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(header):
        values = np.full(len(rows), np.nan)
        for k, row in enumerate(rows):
            if i < len(row):
                try:
                    values[k] = float(row[i])
                except ValueError:
                    pass
        out[name] = values
    return out


def _column(table: dict[str, np.ndarray], name: str, file_name: str) -> np.ndarray:
    if name not in table:
        raise SessionError(
            f"{file_name} has no column named {name}. Columns found: {', '.join(table)}"
        )
    values = table[name]
    if np.any(np.isnan(values)):
        n = int(np.count_nonzero(np.isnan(values)))
        raise SessionError(f"{file_name}: column {name} has {n} empty or non-numeric cells")
    return values


def _write_csv(path: Path, header: list[str], cols: list[np.ndarray]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in zip(*cols):
            writer.writerow([f"{v:.6f}" for v in row])
