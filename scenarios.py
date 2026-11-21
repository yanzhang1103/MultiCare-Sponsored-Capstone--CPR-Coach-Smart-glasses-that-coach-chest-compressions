"""Named test scenarios, shared by the demo and the tests."""

# 30 compressions then a 4-second gap for two breaths, repeated.
_CYCLE_PAUSES = tuple((s, 4.0) for s in (17.0, 38.0, 59.0, 80.0, 101.0))

SCENARIOS = {
    "steady 110": dict(rate_cpm=110),
    "too slow 85": dict(rate_cpm=85),
    "too fast 140": dict(rate_cpm=140),
    "tiring: 115 down to 95": dict(rate_cpm=115, rate_drift_cpm=-20),
    "pauses for breaths": dict(rate_cpm=110, pauses=_CYCLE_PAUSES),
    "shallow compressions": dict(rate_cpm=110, depth_m=0.03),
    "noisy sensor, more sway": dict(rate_cpm=110, noise_m=0.006, sway_m=0.03),
    "uneven rhythm": dict(rate_cpm=110, rate_jitter=0.10, depth_jitter=0.25),
    "head glances": dict(rate_cpm=110, n_glances=10),
    "low sample rate (30 Hz)": dict(rate_cpm=110, sample_hz=30),
    "everything at once": dict(
        rate_cpm=125,
        rate_drift_cpm=-20,
        rate_jitter=0.08,
        depth_jitter=0.2,
        pauses=_CYCLE_PAUSES,
        noise_m=0.005,
        sway_m=0.03,
        n_glances=10,
    ),
}
