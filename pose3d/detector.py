"""
3D pose detector using MediaPipe Pose Landmarker (Tasks API).

Pretrained model used
---------------------
MediaPipe Pose Landmarker **full** (~7 MB, default).
Three model variants are available:
  - 0  lite    ~3 MB   fastest, lower accuracy
  - 1  full    ~7 MB   good accuracy / speed balance  ← default
  - 2  heavy   ~26 MB  highest accuracy, slowest

On first use the appropriate ``.task`` bundle is downloaded automatically to
``~/.cache/mediapipe/`` and cached for subsequent runs.

The model returns 33 body landmarks in both
  * image-normalised coordinates (x, y, z)  – z encodes depth relative to
    the mid-hip point in image space, and
  * metric world coordinates (x_world, y_world, z_world) in metres relative
    to the body centre.
This module exposes the world-space coordinates as the primary 3D output
because they are scale-consistent across frames and suitable for animation
and biomechanics analysis.

Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ── model download URLs (Google's public CDN) ─────────────────────────────────
_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
       "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

_MODEL_NAMES = {0: "lite (~3 MB)", 1: "full (~7 MB)", 2: "heavy (~26 MB)"}

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")


def _get_model_path(complexity: int) -> str:
    """Return the local path to the .task model, downloading it if needed."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    filename = os.path.basename(_MODEL_URLS[complexity])
    local_path = os.path.join(_CACHE_DIR, filename)
    if not os.path.isfile(local_path):
        print(f"[INFO] Downloading MediaPipe Pose Landmarker {_MODEL_NAMES[complexity]} …")
        urllib.request.urlretrieve(_MODEL_URLS[complexity], local_path)
        print(f"[INFO] Model cached at {local_path}")
    return local_path


# ── landmark indices ──────────────────────────────────────────────────────────
LANDMARK_NAMES: List[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Skeleton edges used for visualisation
SKELETON_EDGES: List[tuple] = [
    # face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # mouth
    (9, 10),
    # shoulders
    (11, 12),
    # left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # torso
    (11, 23), (12, 24), (23, 24),
    # left leg
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # right leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


@dataclass
class PoseResult:
    """Stores the 3-D pose for a single frame."""

    frame_index: int
    keypoints_3d: np.ndarray          # (33, 3) world-space (x, y, z) metres
    visibility: np.ndarray            # (33,)  per-landmark confidence [0, 1]
    keypoints_image: Optional[np.ndarray] = None  # (33, 3) image-normalised

    @property
    def landmark_names(self) -> List[str]:
        return LANDMARK_NAMES

    def to_dict(self) -> dict:
        """Serialisable representation of the result."""
        return {
            "frame_index": self.frame_index,
            "keypoints_3d": self.keypoints_3d.tolist(),
            "visibility": self.visibility.tolist(),
            "landmark_names": self.landmark_names,
        }


class PoseDetector3D:
    """
    Detects 3-D human body pose in images / video frames.

    Uses the MediaPipe Pose Landmarker Tasks API (mediapipe >= 0.10).
    The model bundle is downloaded automatically on first use.

    Parameters
    ----------
    model_complexity : int
        0 = lite (~3 MB), 1 = full (~7 MB, default), 2 = heavy (~26 MB).
    min_detection_confidence : float
        Minimum confidence value for the person-detection model.
    min_tracking_confidence : float
        Minimum confidence for landmark tracking.
    model_path : str or None
        Path to a local ``.task`` model file.  When *None* (default) the
        appropriate bundle is downloaded automatically and cached under
        ``~/.cache/mediapipe/``.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = None,
    ) -> None:
        if model_complexity not in (0, 1, 2):
            raise ValueError("model_complexity must be 0, 1, or 2.")
        self.model_complexity = model_complexity

        resolved_path = model_path or _get_model_path(model_complexity)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int = 0) -> Optional[PoseResult]:
        """
        Run pose estimation on a single BGR image (NumPy array).

        Returns ``None`` when no person is detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection = self._landmarker.detect(mp_image)

        if not detection.pose_world_landmarks:
            return None

        world_lms = detection.pose_world_landmarks[0]
        image_lms = detection.pose_landmarks[0] if detection.pose_landmarks else None

        keypoints_3d = np.array(
            [[lm.x, lm.y, lm.z] for lm in world_lms], dtype=np.float32
        )
        visibility = np.array(
            [lm.visibility for lm in world_lms], dtype=np.float32
        )

        keypoints_image = None
        if image_lms is not None:
            keypoints_image = np.array(
                [[lm.x, lm.y, lm.z] for lm in image_lms], dtype=np.float32
            )

        return PoseResult(
            frame_index=frame_index,
            keypoints_3d=keypoints_3d,
            visibility=visibility,
            keypoints_image=keypoints_image,
        )

    def process_frames(self, frames: List[np.ndarray]) -> List[Optional[PoseResult]]:
        """Run pose estimation on a list of BGR frames."""
        return [self.process_frame(f, i) for i, f in enumerate(frames)]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    # ── context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "PoseDetector3D":
        return self

    def __exit__(self, *_) -> None:
        self.close()
