"""CPR compression engine: measurements from the glasses' sensor data."""

from .align import Alignment, align_events, align_session
from .coach import PROMPTS, Coach, CoachConfig, CoachState, run_live
from .fatigue import FatigueConfig, FatigueMonitor
from .depth import Calibration, fit_calibration, head_travel
from .rate import detect_compressions, find_pauses, rate_in_window, rate_series, run_mask
from .scorecard import format_scorecard, pair_session, score_session
from .scoring import (
    depth_band,
    match_events,
    pair_events,
    score_depth,
    score_pauses,
    score_rate,
)
from .session import (
    Session,
    SessionError,
    load_calibration,
    load_session,
    save_calibration,
    save_session,
)
from .stream import Compression, StreamingDetector
from .simulate import SimSession, simulate_manikin_clock, simulate_session

__all__ = [
    "FatigueConfig",
    "FatigueMonitor",
    "PROMPTS",
    "Coach",
    "CoachConfig",
    "CoachState",
    "run_live",
    "Compression",
    "StreamingDetector",
    "Session",
    "SessionError",
    "load_calibration",
    "load_session",
    "save_calibration",
    "save_session",
    "format_scorecard",
    "pair_session",
    "score_session",
    "run_mask",
    "Alignment",
    "align_events",
    "align_session",
    "simulate_manikin_clock",
    "Calibration",
    "fit_calibration",
    "head_travel",
    "detect_compressions",
    "find_pauses",
    "rate_in_window",
    "rate_series",
    "depth_band",
    "match_events",
    "pair_events",
    "score_depth",
    "score_pauses",
    "score_rate",
    "SimSession",
    "simulate_session",
]
