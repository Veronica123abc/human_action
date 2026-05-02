# human_action
Assess biomechanics of human activities

## Overview

`human_action` generates **3D representations of human actions** from:
- a **video file** (`.mp4`, `.avi`, `.mov`, …)
- a **single image file** (`.jpg`, `.png`, …)
- an **image-sequence directory** (folder of numbered frames)

The pipeline uses **MediaPipe Pose Landmarker** (BlazePose backbone) to detect 33
body landmarks in metric world space (x, y, z in metres) and renders them as
colour-coded 3D skeleton plots.

### Pretrained Model

| Variant | Size  | Complexity flag | Notes |
|---------|-------|-----------------|-------|
| **lite**  | ~3 MB | `--model-complexity 0` | Fastest, lower accuracy |
| **full**  | ~7 MB | `--model-complexity 1` | **Default** – good balance |
| **heavy** | ~26 MB | `--model-complexity 2` | Highest accuracy, slowest |

The model bundle is **downloaded automatically** on first run and cached under
`~/.cache/mediapipe/`.

---

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

Python ≥ 3.8 is required.

---

## Quick Start

### Single image

```bash
python main.py --input photo.jpg --output output/
```

### Video file

```bash
python main.py --input walk.mp4 --output output/ --fps 25
```

### Image sequence (directory)

```bash
python main.py --input frames/ --output output/
```

---

## CLI Reference

```
python main.py --input <source> [options]

Required:
  --input  PATH         Video file, image file, or directory of images.

Optional:
  --output DIR          Output directory (default: output/).
  --model-complexity N  0=lite, 1=full (default), 2=heavy.
  --min-detection-confidence FLOAT
                        Person detector threshold (default: 0.5).
  --min-tracking-confidence FLOAT
                        Tracking threshold (default: 0.5).
  --skip-frames N       Skip N frames between processed frames (default: 0).
  --max-frames N        Max frames to process, 0=all (default: 0).
  --fps FLOAT           Output animation FPS (0=auto-detect, default: 0).
  --animate             Save an animated MP4/GIF of the skeleton sequence.
  --save-json           Export 3D keypoints to keypoints3d.json.
  --trajectory-landmarks IDX [IDX ...]
                        Plot 3D trajectories for these landmark indices.
                        E.g. --trajectory-landmarks 15 16 (wrists).
  --vis-threshold FLOAT Minimum landmark visibility to render (default: 0.3).
```

### Example with all options

```bash
python main.py \
  --input workout.mp4 \
  --output results/ \
  --model-complexity 1 \
  --skip-frames 1 \
  --animate \
  --save-json \
  --trajectory-landmarks 15 16 23 24
```

---

## Output Structure

```
output/
├── frames/
│   ├── pose3d_00000.png   # 3D skeleton plot per frame
│   ├── pose3d_00001.png
│   └── …
├── animation.mp4          # (or .gif if ffmpeg unavailable)  --animate
├── trajectory.png         # joint trajectory plot            --trajectory-landmarks
└── keypoints3d.json       # 3D coordinates (JSON)            --save-json
```

### `keypoints3d.json` format

```json
[
  {
    "frame_index": 0,
    "landmark_names": ["nose", "left_eye_inner", …],
    "keypoints_3d": [[x, y, z], …],   // 33 × 3, metres, world space
    "visibility": [0.98, 0.95, …]      // 33 confidence values [0, 1]
  },
  …
]
```

---

## Python API

```python
from pose3d import PoseDetector3D, Visualizer3D, load_frames

frames = load_frames("walk.mp4", skip_frames=1)

with PoseDetector3D(model_complexity=1) as detector:
    results = detector.process_frames(frames)

viz = Visualizer3D()
for result in results:
    if result is not None:
        viz.plot_frame(result, f"output/frame_{result.frame_index:05d}.png")

viz.animate_sequence(results, "output/animation.gif", fps=15)
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Landmark Index Reference

| Index | Name           | Index | Name            |
|-------|----------------|-------|-----------------|
| 0     | nose           | 17    | left_pinky      |
| 11    | left_shoulder  | 18    | right_pinky     |
| 12    | right_shoulder | 23    | left_hip        |
| 13    | left_elbow     | 24    | right_hip       |
| 14    | right_elbow    | 25    | left_knee       |
| 15    | left_wrist     | 26    | right_knee      |
| 16    | right_wrist    | 27    | left_ankle      |
|       |                | 28    | right_ankle     |

Full list of all 33 landmarks: `pose3d.detector.LANDMARK_NAMES`
