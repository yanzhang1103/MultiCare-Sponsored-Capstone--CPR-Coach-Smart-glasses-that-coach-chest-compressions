"""Coaching rules: turn live compressions into what to show and say.

The detector (stream.py) says when each compression happened and how
far the head moved. This file decides what the person is told:

- a rate cue:  "faster", "slower", "good", or nothing
- a depth cue: "push harder", "ease off", "good", or nothing
- a spoken prompt now and then, never a stream of them

It also runs a metronome: a steady tick at the target tempo, so the
person has an exact beat to match instead of guessing from "a little
faster". In the default "adaptive" mode it ticks from the start, falls
silent once the person has held the right rate for a while, and comes
back if they drift. Spoken rate prompts then become a last resort.

It also watches for fatigue (fatigue.py): a "tiring" flag when the
person's compressions fade against their own start, and a reminder to
swap after two minutes.

Three rules shape it (decision log E3, E12, E22):

1. Stay quiet when unsure. "Good" is shown only when the reading is a
   margin inside the target range. Between the edge of the range and
   that margin, nothing is shown.
2. Do not flicker. A cue must hold for a moment before it appears.
   "Good" is the exception in the other direction: it is withdrawn at
   once when the reading leaves the safe zone.
3. Do not nag. Spoken prompts are spaced out, and a correction is
   repeated only after a while.

Plain arithmetic and small fixed-size lists, so it translates to C#
line for line.

Use:

    coach = Coach(calibration)        # calibration may be None: no depth cue
    for t, height in samples:
        for c in detector.update(t, height):
            coach.on_compression(c)
        state = coach.tick(t)         # what to show right now
        if state.prompt:
            speak(state.prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .depth import Calibration
from .fatigue import FatigueConfig, FatigueMonitor
from .stream import Compression

NAN = float("nan")


@dataclass
class CoachConfig:
    good_rate_cpm: tuple[float, float] = (100.0, 120.0)
    rate_margin_cpm: float = 3.0        # "good" only this far inside the range
    good_depth_m: tuple[float, float] = (0.05, 0.06)
    depth_margin_m: float = 0.002       # "good" only this far inside the range
    ease_off_margin_m: float = 0.002    # "ease off" only this far past the deep end

    rate_over_n: int = 8                # rate = median of this many recent gaps
    rate_min_n: int = 4                 # no rate until this many gaps are seen
    depth_over_n: int = 7               # depth = median of this many recent compressions
    depth_min_n: int = 3

    hold_s: float = 1.5                 # a cue must hold this long before it shows
    pause_s: float = 2.0                # no compression for this long = paused
    hands_off_alert_s: float = 10.0     # prompt to restart after this long paused

    prompt_gap_s: float = 5.0           # at least this long between any two prompts
    repeat_s: float = 12.0              # repeat an uncorrected problem this often
    praise_gap_s: float = 20.0          # at most this often

    metronome_mode: str = "adaptive"    # "adaptive", "always" or "off"
    metronome_bpm: float = 110.0        # ticks per minute: the middle of the target range
    metronome_settle_s: float = 10.0    # adaptive: silent after holding the right rate this long
    metronome_grace_s: float = 6.0      # with the tick on, wait this long before speaking about rate

    fatigue_enabled: bool = True
    fatigue_repeat_s: float = 30.0      # repeat the tiring prompt this often while it lasts
    fatigue: FatigueConfig = field(default_factory=FatigueConfig)


PROMPTS = {
    "faster": "A little faster",
    "slower": "A little slower",
    "push harder": "Push harder",
    "ease off": "Not so deep",
    "hands off": "Start compressions again",
    "praise": "Good, keep that up",
    "tiring": "You're tiring. Push harder, or swap if you can",
    "swap": "Two minutes. Swap if someone can take over",
}


@dataclass
class CoachState:
    t: float
    active: bool            # compressions are under way
    rate_cpm: float         # NaN when there is no reading
    rate_cue: str           # "", "faster", "slower" or "good"
    depth_m: float          # NaN when there is no reading or no calibration
    depth_cue: str          # "", "push harder", "ease off" or "good"
    hands_off_s: float      # time since the last compression (0 before the first)
    prompt: str | None      # words to speak now; None nearly all the time
    tiring: bool = False    # compressions have faded against the person's own start
    decline: float = NAN    # by what share (NaN until the first 30 seconds are in)
    metronome_on: bool = False     # the tick should be sounding
    metronome_tick: bool = False   # play one tick now (true once per beat while on)


class _Cue:
    """One cue with the hold rule: what it shows now, and what is waiting to show."""

    def __init__(self):
        self.shown = ""
        self._waiting = ""
        self._since = 0.0

    def update(self, wanted: str, now: float, hold_s: float) -> str:
        if wanted == self.shown:
            self._waiting = wanted
        elif self.shown == "good":
            # Never leave "good" up while the reading says otherwise.
            self.shown = ""
            self._waiting, self._since = wanted, now
        elif wanted != self._waiting:
            self._waiting, self._since = wanted, now
        elif wanted == "" or now - self._since >= hold_s:
            self.shown = wanted
        return self.shown

    def clear(self) -> None:
        self.shown = ""
        self._waiting = ""


class _Metronome:
    """When the tick sounds, and when each beat falls."""

    def __init__(self, cfg: CoachConfig):
        if cfg.metronome_mode not in ("adaptive", "always", "off"):
            raise ValueError('metronome_mode must be "adaptive", "always" or "off"')
        self.mode = cfg.metronome_mode
        self.period = 60.0 / cfg.metronome_bpm
        self._settled = False               # adaptive: the person has the rate, tick is silent
        self._in_range_since: float | None = None
        self._next_beat: float | None = None

    def update(self, now, started, hands_off, rate, rate_cue, cfg) -> tuple[bool, bool]:
        """Returns (on, tick): whether the tick is sounding, and whether a beat falls now."""
        if self.mode == "off":
            return False, False

        if started and hands_off >= cfg.hands_off_alert_s:
            # A long stop: fall silent, and start afresh when they come back.
            on = False
            self._settled = False
            self._in_range_since = None
        elif self.mode == "always":
            on = True
        else:
            lo, hi = cfg.good_rate_cpm
            if self._settled:
                if rate_cue in ("faster", "slower"):
                    self._settled = False           # drifted: bring the tick back
                    self._in_range_since = None
            elif rate == rate:                      # there is a reading
                if lo <= rate <= hi:
                    if self._in_range_since is None:
                        self._in_range_since = now
                    elif now - self._in_range_since >= cfg.metronome_settle_s:
                        self._settled = True
                else:
                    self._in_range_since = None
            on = not self._settled

        if not on:
            self._next_beat = None
            return False, False
        if self._next_beat is None:
            self._next_beat = now
        tick = now >= self._next_beat
        if tick:
            self._next_beat += self.period
            if self._next_beat <= now:              # fell behind: do not burst to catch up
                self._next_beat = now + self.period
        return True, tick


def _median(values) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class Coach:
    def __init__(self, calibration: Calibration | None = None, config: CoachConfig | None = None):
        self.calibration = calibration
        self.cfg = config or CoachConfig()
        self._times: list[float] = []       # recent compression times
        self._travel: list[float] = []      # recent head travel, steadied where possible
        self._last_t: float | None = None
        self._rate_cue = _Cue()
        self._depth_cue = _Cue()
        self._last_prompt_t = -1e9
        self._last_said: dict[str, float] = {}      # when each kind of prompt was last spoken
        self._corrected_since_praise = False
        self._hands_off_said = False
        self._fatigue = FatigueMonitor(self.cfg.fatigue) if self.cfg.fatigue_enabled else None
        self._tiring_said = False
        self._wanted = {"rate": "", "depth": ""}    # what the latest readings call for
        self._metronome = _Metronome(self.cfg)
        self._rate_fix_since: float | None = None   # when the current faster/slower cue appeared

    # ---- inputs -----------------------------------------------------------------

    def on_compression(self, c: Compression) -> None:
        cfg = self.cfg
        if self._last_t is not None and c.t - self._last_t > cfg.pause_s:
            # A new run after a pause: old readings no longer describe it.
            self._times = []
            self._travel = []
        elif self._travel and c.prev_rise_m == c.prev_rise_m:   # not NaN
            # Steady the previous compression's travel with how far the head
            # came back up after it. This cancels slow posture sway.
            self._travel[-1] = 0.5 * (self._travel[-1] + c.prev_rise_m)

        self._times.append(c.t)
        if len(self._times) > cfg.rate_over_n + 1:
            self._times.pop(0)
        if c.travel_m == c.travel_m:
            self._travel.append(c.travel_m)
            if len(self._travel) > cfg.depth_over_n:
                self._travel.pop(0)
        self._last_t = c.t
        self._hands_off_said = False
        if self._fatigue is not None:
            self._fatigue.on_compression(c.t, c.travel_m)

    def tick(self, now: float) -> CoachState:
        """What to show at time `now`. Call it every frame."""
        cfg = self.cfg
        hands_off = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        active = self._last_t is not None and hands_off <= cfg.pause_s

        rate = depth = NAN
        if active:
            rate = self._rate()
            depth = self._depth()
            self._wanted = {"rate": self._want_rate(rate), "depth": self._want_depth(depth)}
            rate_cue = self._rate_cue.update(self._wanted["rate"], now, cfg.hold_s)
            depth_cue = self._depth_cue.update(self._wanted["depth"], now, cfg.hold_s)
        else:
            self._wanted = {"rate": "", "depth": ""}
            self._rate_cue.clear()
            self._depth_cue.clear()
            rate_cue = depth_cue = ""

        # Tiring only counts while the person is not too deep: easing off
        # from too deep is what they were asked to do.
        tiring = False
        decline = NAN
        if self._fatigue is not None:
            decline = self._fatigue.decline
            too_deep = depth == depth and depth > cfg.good_depth_m[1]
            tiring = active and self._fatigue.tiring and depth_cue != "ease off" and not too_deep
            if not self._fatigue.tiring:
                self._tiring_said = False

        if rate_cue in ("faster", "slower"):
            if self._rate_fix_since is None:
                self._rate_fix_since = now
        else:
            self._rate_fix_since = None
        metronome_on, metronome_tick = self._metronome.update(
            now, self._last_t is not None, hands_off, rate, rate_cue, cfg
        )

        prompt = self._choose_prompt(
            now, active, hands_off, rate_cue, depth_cue, tiring, metronome_on
        )
        return CoachState(
            now, active, rate, rate_cue, depth, depth_cue, hands_off, prompt, tiring, decline,
            metronome_on, metronome_tick,
        )

    # ---- readings ---------------------------------------------------------------

    def _rate(self) -> float:
        gaps = [b - a for a, b in zip(self._times[:-1], self._times[1:])]
        if len(gaps) < self.cfg.rate_min_n:
            return NAN
        return 60.0 / _median(gaps)

    def _depth(self) -> float:
        if self.calibration is None or len(self._travel) < self.cfg.depth_min_n:
            return NAN
        return self.calibration.gain * _median(self._travel) + self.calibration.offset_m

    def _want_rate(self, rate: float) -> str:
        if rate != rate:
            return ""
        lo, hi = self.cfg.good_rate_cpm
        m = self.cfg.rate_margin_cpm
        if rate < lo:
            return "faster"
        if rate > hi:
            return "slower"
        if lo + m <= rate <= hi - m:
            return "good"
        return ""

    def _want_depth(self, depth: float) -> str:
        if depth != depth:
            return ""
        lo, hi = self.cfg.good_depth_m
        m = self.cfg.depth_margin_m
        if depth < lo:
            return "push harder"
        if depth > hi + self.cfg.ease_off_margin_m:
            # Shallow compressions are the common fault, so the coach is
            # slower to tell someone to ease off than to push harder.
            return "ease off"
        if lo + m <= depth <= hi - m:
            return "good"
        return ""

    # ---- speech -----------------------------------------------------------------

    def _choose_prompt(
        self, now, active, hands_off, rate_cue, depth_cue, tiring, metronome_on=False
    ) -> str | None:
        cfg = self.cfg
        if now - self._last_prompt_t < cfg.prompt_gap_s:
            return None

        if not active:
            if (
                self._last_t is not None
                and hands_off >= cfg.hands_off_alert_s
                and not self._hands_off_said
            ):
                self._hands_off_said = True
                return self._say("hands off", now)
            return None

        # Corrections: spoken once the cue shows, and again only after a
        # while, even if the cue has flickered off and on in between. If
        # both are due, the one unsaid longest goes first, depth before rate
        # on a tie.
        due = []
        for kind, cue in (("depth", depth_cue), ("rate", rate_cue)):
            if cue in ("", "good"):
                continue
            if self._wanted[kind] != cue:
                # The latest reading no longer calls for this correction; the
                # cue on screen is about to change. Do not say stale words.
                continue
            if (
                kind == "rate"
                and metronome_on
                and now - (self._rate_fix_since or now) < cfg.metronome_grace_s
            ):
                # The tick is already showing the tempo: give it a moment
                # to work before speaking about rate.
                continue
            last = self._last_said.get(cue, -1e9)
            if now - last >= cfg.repeat_s:
                due.append((last, kind, cue))

        # Tiring is said once when it starts and repeated now and then. It
        # already says "push harder", so it stands in for that correction.
        if tiring and (
            not self._tiring_said
            or now - self._last_said.get("tiring", -1e9) >= cfg.fatigue_repeat_s
        ):
            self._tiring_said = True
            self._corrected_since_praise = True
            if depth_cue == "push harder":
                self._last_said["push harder"] = now
            return self._say("tiring", now)

        if due:
            due.sort(key=lambda d: d[0])
            _, _, cue = due[0]
            self._corrected_since_praise = True
            if cue == "ease off" and self._fatigue is not None:
                # Getting shallower on request is not fatigue: measure afresh.
                self._fatigue.rebaseline()
            return self._say(cue, now)

        if self._fatigue is not None and self._fatigue.swap_due:
            self._fatigue.swap_said()
            return self._say("swap", now)

        # Praise: once everything measured is good after a correction.
        depth_fine = depth_cue == "good" or self.calibration is None
        if (
            self._corrected_since_praise
            and rate_cue == "good"
            and depth_fine
            and now - self._last_said.get("praise", -1e9) >= cfg.praise_gap_s
        ):
            self._corrected_since_praise = False
            return self._say("praise", now)
        return None

    def _say(self, key: str, now: float) -> str:
        self._last_prompt_t = now
        self._last_said[key] = now
        return PROMPTS[key]


def run_live(t, y, calibration: Calibration | None = None, config: CoachConfig | None = None):
    """Run the live detector and coach over a whole recording, as if it were live.

    Returns (compressions, states): every compression the detector
    reported, and the coach's state at every sample.
    """
    from .stream import StreamingDetector

    detector = StreamingDetector()
    coach = Coach(calibration, config)
    compressions: list[Compression] = []
    states: list[CoachState] = []
    for ti, yi in zip(t, y):
        ti = float(ti)
        for c in detector.update(ti, float(yi)):
            coach.on_compression(c)
            compressions.append(c)
        states.append(coach.tick(ti))
    return compressions, states
