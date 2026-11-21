r"""Print the scorecard for one or more recorded sessions.

    python score_session.py sessions\sample_P1_test
    python score_session.py sessions\sample_P1_test --calibration calibration.json

Without --calibration, rate and pauses are scored and depth is not.
"""

from __future__ import annotations

import argparse
import sys

from cpr_engine import (
    SessionError,
    format_scorecard,
    load_calibration,
    load_session,
    score_session,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+", help="session folder(s)")
    parser.add_argument("--calibration", help="calibration file from fit_calibration.py")
    args = parser.parse_args()

    failed = 0
    try:
        calibration = load_calibration(args.calibration) if args.calibration else None
    except SessionError as e:
        print(f"ERROR  {e}")
        return 1

    for i, folder in enumerate(args.sessions):
        if i:
            print("\n" + "=" * 78 + "\n")
        try:
            session = load_session(folder)
        except SessionError as e:
            print(f"Session {folder}\n  ERROR  {e}")
            failed += 1
            continue
        print(format_scorecard(score_session(session, calibration)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
