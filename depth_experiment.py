"""The depth experiment: does one calibration hold across people?

Each person records a calibration sweep (shallow to deep) and three
test sessions (shallow, good, deep). Two ways of calibrating are
compared on the test sessions:

- one for everyone: fitted on all the *other* people's sweeps, so the
  person being tested is new to it (leave-one-person-out).
- per person: fitted on that person's own sweep.

The same functions will run on real recordings: replace
`simulate_person` with a loader that returns the same structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpr_engine import (
    detect_compressions,
    fit_calibration,
    head_travel,
    pair_events,
    simulate_session,
)

TEST_DEPTHS_M = (0.042, 0.055, 0.066)   # shallow, good, deep


@dataclass
class PersonData:
    name: str
    head_gain: float            # known only in simulation
    sweep_travel: np.ndarray    # calibration sweep: head travel per compression
    sweep_depth: np.ndarray     # calibration sweep: true depth per compression
    test_travel: np.ndarray     # test sessions, pooled
    test_depth: np.ndarray


def paired_travel_and_depth(session) -> tuple[np.ndarray, np.ndarray]:
    """Detect compressions, measure head travel, pair with the answer key."""
    est = detect_compressions(session.t, session.y)
    travel = head_travel(session.t, session.y, est)
    est_idx, true_idx = pair_events(est, session.true_times)
    return travel[est_idx], session.true_depths_m[true_idx]


def simulate_person(
    name: str,
    head_gain: float,
    seed: int,
    coupling_jitter: float = 0.04,
    session_s: float = 60.0,
) -> PersonData:
    sweep = simulate_session(
        duration_s=session_s,
        depth_m=0.035,
        depth_drift_m=0.035,
        depth_jitter=0.05,
        head_gain=head_gain,
        head_coupling_jitter=coupling_jitter,
        seed=seed,
    )
    sweep_travel, sweep_depth = paired_travel_and_depth(sweep)

    travels, depths = [], []
    for k, depth in enumerate(TEST_DEPTHS_M):
        s = simulate_session(
            duration_s=session_s,
            depth_m=depth,
            head_gain=head_gain,
            head_coupling_jitter=coupling_jitter,
            seed=seed + 1 + k,
        )
        tr, d = paired_travel_and_depth(s)
        travels.append(tr)
        depths.append(d)

    return PersonData(
        name=name,
        head_gain=head_gain,
        sweep_travel=sweep_travel,
        sweep_depth=sweep_depth,
        test_travel=np.concatenate(travels),
        test_depth=np.concatenate(depths),
    )


def simulate_people(
    n_people: int = 6,
    mean_gain: float = 0.9,
    gain_spread: float = 0.08,
    seed: int = 0,
    **kwargs,
) -> list[PersonData]:
    """gain_spread: person-to-person variation in head gain, as a fraction (1 SD)."""
    rng = np.random.default_rng(seed)
    gains = mean_gain * (1.0 + gain_spread * rng.standard_normal(n_people))
    return [
        simulate_person(f"P{i + 1}", float(g), seed=1000 * (seed + 1) + 10 * i, **kwargs)
        for i, g in enumerate(gains)
    ]


def estimate_one_for_everyone(people: list[PersonData]) -> list[np.ndarray]:
    """Depth estimates for each person from a calibration fitted without them."""
    out = []
    for i, person in enumerate(people):
        others = [p for j, p in enumerate(people) if j != i]
        cal = fit_calibration(
            np.concatenate([p.sweep_travel for p in others]),
            np.concatenate([p.sweep_depth for p in others]),
        )
        out.append(cal.depth(person.test_travel))
    return out


def estimate_per_person(people: list[PersonData]) -> list[np.ndarray]:
    """Depth estimates for each person from their own calibration sweep."""
    return [
        fit_calibration(p.sweep_travel, p.sweep_depth).depth(p.test_travel)
        for p in people
    ]
