"""Simulated head-motion sessions with a known answer.

Stands in for real recordings until the glasses and manikin arrive.
It proves the code works; it does not prove that real head motion
looks like this. The sizes below (head travel, sway, noise) are
guesses to be replaced by what the first recordings show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimSession:
    t: np.ndarray              # sample times, seconds
    y: np.ndarray              # head height, metres (0 = resting height)
    true_times: np.ndarray     # time of the bottom of each compression, seconds
    true_depths_m: np.ndarray  # chest depth of each compression, metres
    duration_s: float


def simulate_session(
    duration_s: float = 120.0,
    rate_cpm: float = 110.0,
    rate_drift_cpm: float = 0.0,
    rate_jitter: float = 0.03,
    depth_m: float = 0.055,
    depth_jitter: float = 0.08,
    depth_drift_m: float = 0.0,
    depth_drift_from_s: float = 0.0,
    head_gain: float = 0.9,
    head_coupling_jitter: float = 0.0,
    pauses: tuple[tuple[float, float], ...] = (),
    start_cue: int = 0,
    cue_interval_s: float = 1.0,
    cue_gap_s: float = 3.0,
    noise_m: float = 0.002,
    sway_m: float = 0.01,
    n_glances: int = 0,
    glance_m: float = 0.03,
    sample_hz: float = 60.0,
    timestamp_jitter: float = 0.2,
    seed: int = 0,
) -> SimSession:
    """Make one simulated session.

    rate_cpm:        starting compression rate, per minute.
    rate_drift_cpm:  change in rate by the end (negative = slowing down).
    rate_jitter:     beat-to-beat variation in timing, as a fraction.
    depth_m:         average chest depth.
    depth_jitter:    compression-to-compression variation in depth, as a fraction.
    depth_drift_m:   change in average depth by the end (calibration sweeps, fatigue).
    depth_drift_from_s: the depth holds steady until this time, then drifts.
    head_gain:       head travel divided by chest travel (the open question).
    head_coupling_jitter: compression-to-compression variation in head travel
                     that chest depth does not explain (bent arms, neck), as a fraction.
    pauses:          (start, length) in seconds where compressions stop.
    start_cue:       number of sharp, slow compressions at the very start, used
                     to line up the glasses and manikin recordings (0 = none).
    cue_interval_s:  time from one cue compression to the next.
    cue_gap_s:       stillness after the cue before normal compressions begin.
    noise_m:         sensor noise in head height.
    sway_m:          slow drift in posture.
    n_glances:       number of head movements that are not compressions.
    glance_m:        size of each of those movements.
    timestamp_jitter: unevenness of sample timing, as a fraction of one sample.
    """
    rng = np.random.default_rng(seed)

    n = int(duration_s * sample_hz)
    t = np.arange(n) / sample_hz
    t = t + rng.uniform(-0.5, 0.5, n) * timestamp_jitter / sample_hz
    y = np.zeros(n)

    pause_spans = sorted((s, s + length) for s, length in pauses)
    true_times: list[float] = []
    true_depths: list[float] = []

    cur = 1.0  # first compression starts one second in

    # Start cue: a few sharp compressions spaced well apart, then stillness.
    cue_stroke_s = 0.5
    for _ in range(start_cue):
        depth = max(0.01, depth_m * (1.0 + depth_jitter * rng.standard_normal()))
        in_cycle = (t >= cur) & (t < cur + cue_stroke_s)
        phase = (t[in_cycle] - cur) / cue_stroke_s
        y[in_cycle] -= head_gain * depth * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        true_times.append(cur + 0.5 * cue_stroke_s)
        true_depths.append(depth)
        cur += cue_interval_s
    if start_cue:
        cur += cue_gap_s - (cue_interval_s - cue_stroke_s)

    while cur < duration_s:
        in_pause = next((p for p in pause_spans if p[0] <= cur < p[1]), None)
        if in_pause is not None:
            cur = in_pause[1]
            continue

        rate_now = rate_cpm + rate_drift_cpm * (cur / duration_s)
        period = (60.0 / rate_now) * (1.0 + rate_jitter * rng.standard_normal())
        period = max(period, 0.2)
        end = cur + period
        if end > duration_s:
            break

        drift_share = max(0.0, cur - depth_drift_from_s) / max(duration_s - depth_drift_from_s, 1e-9)
        depth_now = depth_m + depth_drift_m * drift_share
        depth = max(0.01, depth_now * (1.0 + depth_jitter * rng.standard_normal()))
        travel = head_gain * depth
        if head_coupling_jitter > 0:
            travel *= 1.0 + head_coupling_jitter * rng.standard_normal()
        in_cycle = (t >= cur) & (t < end)
        phase = (t[in_cycle] - cur) / period
        y[in_cycle] -= travel * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))

        true_times.append(cur + 0.5 * period)
        true_depths.append(depth)
        cur = end

    # Slow posture sway: a few slow waves added together.
    for _ in range(4):
        f = rng.uniform(0.03, 0.3)
        y += (sway_m / 2.0) * np.sin(2.0 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    # Glances: brief head movements unrelated to compressions.
    for _ in range(n_glances):
        centre = rng.uniform(2.0, duration_s - 2.0)
        sign = rng.choice([-1.0, 1.0])
        y += sign * glance_m * np.exp(-0.5 * ((t - centre) / 0.3) ** 2)

    y += noise_m * rng.standard_normal(n)

    return SimSession(
        t=t,
        y=y,
        true_times=np.array(true_times),
        true_depths_m=np.array(true_depths),
        duration_s=duration_s,
    )


def simulate_manikin_clock(
    true_times,
    offset_s: float = 0.0,
    drift_ppm: float = 0.0,
    timing_noise_s: float = 0.003,
    drop_fraction: float = 0.0,
    starts_at_s: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """The same compressions as the manikin's own clock would stamp them.

    The two devices do not share a clock, so:
        glasses time = manikin time x (1 + drift) + offset

    offset_s:       how far the glasses clock is ahead of the manikin clock
                    (at manikin time zero).
    drift_ppm:      how much faster the glasses clock runs, parts per million.
    timing_noise_s: scatter in the manikin's time stamps.
    drop_fraction:  share of compressions the manikin fails to log.
    starts_at_s:    the manikin starts logging at this glasses time
                    (earlier compressions are lost), or None for all.
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(true_times, dtype=float)
    if starts_at_s is not None:
        times = times[times >= starts_at_s]
    if drop_fraction > 0:
        times = times[rng.random(times.size) >= drop_fraction]
    manikin = (times - offset_s) / (1.0 + drift_ppm * 1e-6)
    manikin = manikin + timing_noise_s * rng.standard_normal(manikin.size)
    return np.sort(manikin)
