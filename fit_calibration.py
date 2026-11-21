r"""Fit a depth calibration from one or more sessions and save it.

    python fit_calibration.py sessions\sample_P1_sweep -o calibration.json

Use sessions recorded on the manikin that cover shallow to deep
compressions. Give several people's sessions to fit one calibration
for everyone, or one person's to fit theirs alone.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from cpr_engine import (
    SessionError,
    fit_calibration,
    load_session,
    pair_session,
    save_calibration,
)

MIN_DEPTH_SPREAD_M = 0.015


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+", help="session folder(s)")
    parser.add_argument("-o", "--output", default="calibration.json")
    args = parser.parse_args()

    travels, depths, used = [], [], []
    for folder in args.sessions:
        try:
            session = load_session(folder)
        except SessionError as e:
            print(f"skipped {folder}: {e}")
            continue
        if not session.has_answer_key or session.manikin_depth_m is None:
            print(f"skipped {session.session_id}: no manikin depth")
            continue
        p = pair_session(session)
        if not p.alignment.ok:
            print(f"skipped {session.session_id}: alignment not trusted ({p.alignment.reason})")
            continue
        travel = p.travel_m[p.est_idx]
        depth = p.true_depth_m[p.true_idx]
        ok = ~np.isnan(travel)
        travels.append(travel[ok])
        depths.append(depth[ok])
        used.append(session.session_id)
        print(f"used    {session.session_id}: {int(ok.sum())} compressions")

    if not used:
        print("ERROR  no usable sessions")
        return 1

    travel = np.concatenate(travels)
    depth = np.concatenate(depths)
    try:
        cal = fit_calibration(travel, depth)
    except ValueError as e:
        print(f"ERROR  {e}")
        return 1
    error_mm = float(np.mean(np.abs(depth - cal.depth(travel))) * 1000.0)
    lo, hi = np.percentile(depth, [5, 95])

    save_calibration(
        args.output,
        cal,
        {
            "fitted_from": used,
            "n_compressions": int(travel.size),
            "fit_error_mm": round(error_mm, 2),
            "depth_range_m": [round(float(lo), 4), round(float(hi), 4)],
        },
    )
    print()
    print(f"depth = {cal.gain:.3f} x head travel {cal.offset_m * 1000:+.1f} mm")
    print(f"fitted on {travel.size} compressions, average error {error_mm:.1f} mm")
    print(f"depths covered: {lo * 100:.1f} to {hi * 100:.1f} cm")
    if hi - lo < MIN_DEPTH_SPREAD_M:
        print("WARNING  the depths barely vary. A calibration needs shallow, good and deep")
        print("         compressions; this one may be wrong outside the range above.")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
