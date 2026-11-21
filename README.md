# CPR coach: compression engine

Python measurement code for the CPR coaching glasses. Phase 1 of the technical plan: build and score the measurements offline before any glasses app exists.

## What is here

| File | What it does |
| --- | --- |
| `cpr_engine/rate.py` | Finds each compression in the head-height signal; gives the rate and the pauses |
| `cpr_engine/depth.py` | Measures head travel per compression; fits the calibration that turns it into chest depth |
| `cpr_engine/align.py` | Lines up the glasses and manikin recordings, which keep separate clocks |
| `cpr_engine/session.py` | Reads and checks a recorded session folder (format in `SESSION_FORMAT.md`) |
| `cpr_engine/scorecard.py` | Scores one session from start to finish |
| `cpr_engine/stream.py` | Live detector: one head-pose sample in, compressions out. The version to translate to C# |
| `cpr_engine/coach.py` | Coaching rules: what to show and say, and when to stay quiet |
| `cpr_engine/fatigue.py` | Fatigue: flags when compressions fade against the person's own start; two-minute swap reminder |
| `cpr_engine/bench.py` | Bench test: pools many sessions, tests calibrations across people, replays the live coach |
| `cpr_engine/live_eval.py` | Checks the live detector and coach against the truth |
| `cpr_engine/marker.py` | Chest marker: finds it in a camera frame, locates the hand target, checks a hand against it |
| `cpr_engine/marker_sim.py` | Generated camera frames of the marker with a known answer |
| `cpr_engine/simulate.py` | Makes simulated sessions with a known answer, until real recordings exist |
| `cpr_engine/scoring.py` | Scores the detector against the answer key using the plan's pass marks |
| `scenarios.py` | Eleven named test scenarios (too slow, tiring, pauses, noisy sensor, and so on) |
| `run_demo.py` | Rate and pauses: prints a score table for every scenario; `--plot` saves a picture |
| `depth_experiment.py` | The depth experiment: one calibration for everyone against one per person |
| `run_depth_demo.py` | Depth: prints the experiment's score tables; `--plot` saves a picture |
| `run_align_demo.py` | Alignment: prints how often it is right, flagged or wrong across many runs |
| `run_fatigue_demo.py` | Fatigue alert: prints how soon it fires on simulated four-minute sessions |
| `run_live_demo.py` | Live detector and coach: prints delay, cue accuracy and prompt counts |
| `make_marker_sheet.py` | Makes the printable chest marker sheet (`chest_marker.pdf`) |
| `run_marker_demo.py` | Marker: tables on generated frames, or finds the marker in a real photo |
| `score_session.py` | Prints the scorecard for a recorded session |
| `bench_report.py` | Scores a whole folder of sessions and prints the bench-test report |
| `fit_calibration.py` | Fits a depth calibration from manikin sessions and saves it |
| `make_sample_sessions.py` | Writes simulated sessions in the real file format |
| `SESSION_FORMAT.md` | The file format the capture app and manikin converter must write |
| `tests/` | Automated tests |

## Set up (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python run_demo.py --plot
python run_depth_demo.py --plot
python run_align_demo.py
python run_live_demo.py
python run_fatigue_demo.py
python run_marker_demo.py
python -m pytest -q
```

## Score a recorded session

```powershell
python make_sample_sessions.py
python fit_calibration.py sessions\sample_P1_sweep -o calibration.json
python score_session.py sessions\sample_P1_test --calibration calibration.json
python score_session.py sessions\sample_P1_pillow --calibration calibration.json
```

The same three commands will run on real recordings once they are saved in the format in `SESSION_FORMAT.md`. A session with no manikin file (pillow practice) gets measurements only.

## Bench-test report

```powershell
python make_sample_sessions.py --bench
python bench_report.py sessions\bench
python bench_report.py sessions\bench --csv bench_sessions.csv
```

Point it at a folder of sessions. Each `session.json` needs `person` (a code such as `P1`) and `kind` (`calibration` for a shallow-to-deep sweep; anything else is a test). The report gives:

- **Measurement:** rate, compressions, pauses and depth against the manikin, pooled over the test sessions, with the plan's pass marks.
- **Depth two ways:** one calibration for everyone (each person scored by a calibration fitted without them) and one per person.
- **Live coach:** every test session is replayed through the live detector and coach, and what the coach showed is checked against the manikin. The "never say good when it is not" guardrail is judged here.
- **By person:** how far each person's head moves per cm of chest depth, how much people differ, and how much one person scatters. These are the two numbers that decide whether one calibration can work.
- **By session**, also as a CSV for the write-up.

Sessions that cannot be read or aligned are listed with the reason and left out; they do not stop the report.

## How the detector works

1. Resample head height to an even 50 samples per second.
2. Filter out slow posture sway and fast sensor noise.
3. Find each dip (the bottom of a compression) that is at least 8 mm deep and at least 40% of the typical dip.
4. Keep only runs of four or more dips in a row, so a glance is not counted as CPR.
5. Rate = 60 divided by the median gap between compressions. A gap over 2 seconds is a pause and is reported separately.

## How depth works

1. For each detected compression, measure head travel: the drop from the top before and the top after to the bottom, averaged.
2. Turn head travel into chest depth with a calibration (depth = gain x travel + offset), fitted against the manikin's depth on a sweep from shallow to deep.
3. The experiment compares one calibration for everyone (tested on a person it has never seen) with a calibration per person.

In simulation, one calibration for everyone meets the proposed 85% band mark only while people differ by about 6% or less in how far the head moves per cm of chest depth. The first real recordings need to measure two numbers: that person-to-person difference, and the compression-to-compression scatter within one person.

## How alignment works

The glasses and the manikin keep separate clocks. Alignment finds the offset between them (and any slow drift) from the compressions themselves, since both devices record the same pattern of compressions and pauses.

1. Slide one recording past the other and note where the compressions line up best.
2. For the best few candidates, pair up the compressions, fine-tune, and score how tightly the pairs agree.
3. Accept the best only if it clearly beats the runner-up. Otherwise the result is flagged with the reason and should not be used.

A very steady rhythm with no cue and no pauses cannot be aligned with confidence: the recordings fit equally well one compression out. That case is flagged, never guessed.

### Recording protocol this implies

- Start both recordings before the first compression.
- Begin with the start cue: three sharp compressions about a second apart, then three seconds of stillness, then normal CPR.
- Repeating the cue at the end is a cheap safeguard for long sessions.

## The live detector and the coach

`stream.py` and `coach.py` are the two files that run on the glasses. They only look backwards in time, use plain arithmetic and small fixed-size lists, and are written to be translated to C# line for line.

```python
detector = StreamingDetector()
coach = Coach(calibration)            # calibration may be None: no depth cue
for t, height in samples:             # as each head-pose sample arrives
    for c in detector.update(t, height):
        coach.on_compression(c)
    state = coach.tick(t)             # rate cue, depth cue, and now and then a prompt to speak
```

What the coach does:

- Shows a rate cue (faster, slower, good) and a depth cue (push harder, ease off, good), or nothing.
- Says "good" only when the reading is a margin inside the target range: 3 per minute for rate, 2 mm for depth. Between the edge of the range and the margin it stays quiet.
- Holds a cue for 1.5 seconds before showing it, and withdraws "good" at once.
- Spaces spoken prompts at least 5 seconds apart and repeats an uncorrected problem every 12 seconds.
- Is slower to say "ease off" (2 mm past the deep end) than "push harder".
- Prompts to restart after 10 seconds without compressions.
- Repeats a correction no sooner than every 12 seconds, even if the cue flickers off and on.
- Never speaks a correction that the latest reading no longer supports.

Metronome (in the coach):

- Ticks at 110 a minute so the person has an exact tempo to match. `CoachState.metronome_on` says whether it should be sounding, and `metronome_tick` is true once per beat; the app plays the sound.
- Three modes in `CoachConfig.metronome_mode`: `adaptive` (default), `always`, `off`.
- Adaptive: ticks from the start, falls silent once the rate has been in range for 10 seconds, and comes back when the rate drifts out. A short pause for breaths does not restart it; a stop of 10 seconds or more does.
- While the tick is on, the coach waits 6 seconds before speaking about rate, so the tick gets a chance to work first.
- Simulated people cannot hear the tick, so the tests show when it sounds, not whether it helps. That is a question for the user test.

Fatigue (`fatigue.py`, run by the coach):

- Compares the median head travel of the last 20 compressions with the person's own first 30 seconds. When it is down 15% for 10 compressions in a row, the coach says "You're tiring. Push harder, or swap if you can", and repeats it every 30 seconds while it lasts.
- Needs no depth calibration, because it compares a person with themselves. It works on a pillow.
- Reminds to swap after two minutes, short pauses included.
- A rest of 30 seconds or more starts it again. Being told "Not so deep" also restarts the baseline, since easing off on request is not fatigue.
- It rests on the same untested idea as depth: that head travel follows chest depth.

All of these numbers are in `CoachConfig` and are first guesses to tune on real sessions.

The detector reports a compression about 120 ms after its bottom, because it has to see the head start to come back up. The first three compressions of a run are held until the fourth arrives. Processing one sample takes microseconds.

For the C# version, `StreamingDetector().filter_coefficients()` prints the filter numbers to copy in as constants. The Python version stays as the reference: run the same recording through both and compare.

## The chest marker

Hand placement needs a fixed reference on the chest. A printed marker is fixed to the manikin a known distance from the correct hand position; finding the marker in a frame gives the target spot in that frame and how far a hand is from it in millimetres.

```powershell
python make_marker_sheet.py                    # chest_marker.pdf; print at 100% scale
python run_marker_demo.py                      # tables on generated frames
python run_marker_demo.py --image photo.jpg    # find it in a real photo, saves photo_marked.jpg
```

- The marker is ArUco 4x4_50, number 7, 80 mm square. OpenCV reads it (for frames from the XREAL Eye camera) and so does Android XR's built-in marker tracking (for the Aura), so one printed marker serves both glasses.
- `MarkerLayout` holds what has to be measured on the real manikin: the offset from the marker's centre to the hand target, and the tolerance for "on target". The defaults (12 cm toward the feet, 3 cm) are placeholders.
- `check_hand` takes the pixel where a hand was seen and says whether it is on target and which way to move. Finding the hand in the frame is separate work and needs real video.
- Two ways to place the hand: the flat method needs only the frame, but reads wrong by hand height times the tangent of the viewing angle (about 17 mm at 20 degrees for a point 40 mm above the chest). The 3D method removes that, and needs the camera's lens numbers (camera matrix).

On generated frames the marker was found at every distance, angle and light level tried, with the target placed within about 1 mm. Two things break it: motion blur past about 16 pixels, and anything covering more than about a fifth of the marker. Keep it clear of the hands, and prefer a short camera exposure. Generated frames have no lens distortion, glare, fabric or curved chest, so real frames will be harder.

## Pass marks used (from the technical plan)

- Rate within ±5 per minute in 90% of 30-second segments.
- Depth (proposed, not yet agreed): the correct band (too shallow, good 5 to 6 cm, too deep) in 85% of compressions.
- Guardrail: the glasses say "good" when the answer key says otherwise in fewer than 5% of cases. For rate, good is 100 to 120 per minute.

## What this does not show yet

- Simulated data proves the code works. It does not prove real head motion looks like this. Head travel, sway and noise sizes in `simulate.py` are guesses.
- The coach's timings and margins are tuned on simulated sessions only.
- The answer key here is the time of every compression. If the Brayden export gives only a summary rate, `scoring.py` needs a second entry point, and alignment is not possible from the export alone.

## Next

- Converter from the Brayden export to `manikin.csv`, once a real export exists.
- Capture app that writes `head_pose.csv` and `session.json`, once the 1 Pro+ arrives.
- C# translation of `stream.py` and `coach.py` for the Unity app.
- Hand detection from video, to feed `check_hand`.
