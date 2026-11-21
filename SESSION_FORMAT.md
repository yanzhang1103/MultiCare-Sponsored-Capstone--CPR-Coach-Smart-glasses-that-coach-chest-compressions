# Session file format (version 1)

One recorded session is one folder. The capture app writes `head_pose.csv` and `session.json`. A converter (to be written once a real Brayden export exists) writes `manikin.csv`.

```
sessions/
  2026-11-24_P1_s01/
    session.json       who, what, when
    head_pose.csv      from the glasses
    manikin.csv        answer key from the manikin; absent for pillow sessions
    video.mp4          Eye camera video (reserved for hand placement; not read yet)
    video_frames.csv   time of each video frame (reserved; not read yet)
```

Run `python make_sample_sessions.py` to see real examples of every file.

## head_pose.csv

One row per head-pose sample. Plain CSV with a header row.

| Column | Required | Unit | Meaning |
| --- | --- | --- | --- |
| `t_s` | yes | seconds | Time of the sample. Must only move forward. Any starting point is fine |
| `x_m`, `y_m`, `z_m` | the up axis is required | metres | Head position in a fixed world frame. Unity's default has `y` up |
| `qx`, `qy`, `qz`, `qw` | no, but save them | none | Head rotation as a quaternion. Not read yet; needed later for hand placement |

Rules for the capture app:

- Use one forward-only clock for everything it records, so pose samples and video frames share a time base. Do not use wall-clock time, which can jump.
- Write `t_s` in seconds with at least millisecond precision.
- 30 samples a second or more. 60 is better.
- Position in metres, not centimetres.
- Start recording before the first compression and stop after the last.

Example:

```
t_s,x_m,y_m,z_m,qx,qy,qz,qw
0.000000,0.000000,0.748695,0.000000,0.000000,0.000000,0.000000,1.000000
0.016667,0.000049,0.747881,-0.000071,0.000000,0.000000,0.000000,1.000000
```

## manikin.csv

One row per compression the manikin recorded. Leave the file out for pillow sessions.

| Column | Required | Unit | Meaning |
| --- | --- | --- | --- |
| `t_s` | yes | seconds | Time of the compression on the manikin's own clock. It does not need to match the glasses clock; alignment finds the difference |
| `depth_m` | no | metres | Depth of that compression. 5 cm is `0.05`. Without it, depth is not scored |

Example:

```
t_s,depth_m
14.049137,0.053828
15.052235,0.055142
```

## session.json

| Key | Required | Meaning |
| --- | --- | --- |
| `format_version` | yes | `1` |
| `session_id` | no | Defaults to the folder name |
| `person` | no | A code such as `P1`, not a name |
| `device` | no | `XREAL 1 Pro+` or `XREAL Aura` |
| `surface` | no | `manikin` or `pillow` |
| `kind` | no | `calibration` (a sweep from shallow to deep), `test` or `practice` |
| `up_axis` | no | `x`, `y` or `z`. Defaults to `y` |
| `recorded_at` | no | Date and time with time zone |
| `notes` | no | Anything worth remembering about the session |

Example:

```json
{
  "format_version": 1,
  "session_id": "2026-11-24_P1_s01",
  "person": "P1",
  "device": "XREAL 1 Pro+",
  "surface": "manikin",
  "kind": "test",
  "up_axis": "y",
  "recorded_at": "2026-11-24T14:05:00-08:00",
  "notes": ""
}
```

## video_frames.csv (reserved)

| Column | Unit | Meaning |
| --- | --- | --- |
| `frame` | none | Frame number in `video.mp4`, from 0 |
| `t_s` | seconds | Time of that frame, on the same clock as `head_pose.csv` |

## What the loader checks

It stops with a plain message when: a file or required column is missing, a cell is empty or not a number, `t_s` looks like milliseconds, time runs backwards, head height is not in metres, or depth is not in metres.

It warns when: the sample rate is under 20 a second, there is a gap in the samples, the session is under 20 seconds, or head height barely changes (often the wrong `up_axis`).

## Recording protocol

1. Start the glasses recording and the manikin session.
2. Start cue: three sharp compressions about a second apart, then three seconds of stillness.
3. Normal CPR.
4. Stop both recordings.

For a calibration session, go from shallow to deep compressions over the session.
