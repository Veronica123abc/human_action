"""
Unit tests for the human_action pose3d package.

Run with:
    pip install -e .[dev]
    pytest tests/
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Allow imports from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Helpers: synthetic frames
# ---------------------------------------------------------------------------

def _blank_frame(height: int = 480, width: int = 640, color=(200, 200, 200)) -> np.ndarray:
    """Return a blank BGR frame (no person – used to test no-detection path)."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    return frame


def _synthetic_keypoints(n: int = 33) -> np.ndarray:
    """Return synthetic (n, 3) world-space keypoints in a rough standing pose."""
    rng = np.random.default_rng(42)
    kp = rng.uniform(-0.5, 0.5, size=(n, 3)).astype(np.float32)
    return kp


def _synthetic_visibility(n: int = 33, threshold: float = 0.9) -> np.ndarray:
    return np.full(n, threshold, dtype=np.float32)


# ---------------------------------------------------------------------------
# io_utils tests
# ---------------------------------------------------------------------------

class TestIoUtils:
    def test_load_single_image(self, tmp_path):
        """load_frames should return one frame for a valid image file."""
        import cv2
        from pose3d.io_utils import load_frames

        img_path = str(tmp_path / "test.png")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, frame)

        frames = load_frames(img_path)
        assert len(frames) == 1
        assert frames[0].shape == (100, 100, 3)

    def test_load_image_sequence(self, tmp_path):
        """load_frames on a directory should return all images sorted by name."""
        import cv2
        from pose3d.io_utils import load_frames

        for i in range(5):
            cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"),
                        np.zeros((50, 50, 3), dtype=np.uint8))

        frames = load_frames(str(tmp_path))
        assert len(frames) == 5

    def test_load_image_sequence_skip_frames(self, tmp_path):
        """skip_frames=1 should return every other frame."""
        import cv2
        from pose3d.io_utils import load_frames

        for i in range(6):
            cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"),
                        np.zeros((50, 50, 3), dtype=np.uint8))

        frames = load_frames(str(tmp_path), skip_frames=1)
        assert len(frames) == 3  # frames 0, 2, 4

    def test_load_image_sequence_max_frames(self, tmp_path):
        """max_frames caps the number of returned frames."""
        import cv2
        from pose3d.io_utils import load_frames

        for i in range(10):
            cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"),
                        np.zeros((50, 50, 3), dtype=np.uint8))

        frames = load_frames(str(tmp_path), max_frames=4)
        assert len(frames) == 4

    def test_load_invalid_path_raises(self):
        from pose3d.io_utils import load_frames

        with pytest.raises((ValueError, IOError)):
            load_frames("/nonexistent/path/file.xyz")

    def test_get_video_fps_for_image(self, tmp_path):
        """get_video_fps should return 1.0 for a plain image path."""
        import cv2
        from pose3d.io_utils import get_video_fps

        img_path = str(tmp_path / "img.png")
        cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype=np.uint8))
        assert get_video_fps(img_path) == 1.0

    def test_iter_frames_single_image(self, tmp_path):
        """iter_frames should yield one frame for a single image."""
        import cv2
        from pose3d.io_utils import iter_frames

        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, np.zeros((64, 64, 3), dtype=np.uint8))
        frames = list(iter_frames(img_path))
        assert len(frames) == 1


# ---------------------------------------------------------------------------
# detector tests (mock the underlying landmarker to avoid network download)
# ---------------------------------------------------------------------------

def _make_mock_landmarker(has_detection: bool = True):
    """Return a mock PoseLandmarker whose detect() method returns fake results."""
    mock_lm = MagicMock()
    mock_lm.x = 0.1
    mock_lm.y = 0.2
    mock_lm.z = 0.3
    mock_lm.visibility = 0.9

    mock_result = MagicMock()
    if has_detection:
        mock_result.pose_world_landmarks = [[mock_lm] * 33]
        mock_result.pose_landmarks = [[mock_lm] * 33]
    else:
        mock_result.pose_world_landmarks = []
        mock_result.pose_landmarks = []

    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = mock_result
    return mock_landmarker


class TestPoseDetector3D:
    def test_instantiation_valid_complexity(self):
        """PoseDetector3D should construct without errors for valid complexities."""
        from pose3d.detector import PoseDetector3D

        # Patch PoseLandmarker to avoid downloading models in tests
        with patch("pose3d.detector.PoseLandmarker") as MockLandmarker, \
             patch("pose3d.detector._get_model_path", return_value="/fake/model.task"):
            MockLandmarker.create_from_options.return_value = MagicMock()
            for c in (0, 1, 2):
                d = PoseDetector3D(model_complexity=c)
                d.close()

    def test_instantiation_invalid_complexity(self):
        """model_complexity outside 0–2 should raise ValueError."""
        from pose3d.detector import PoseDetector3D

        with pytest.raises(ValueError):
            PoseDetector3D(model_complexity=3)

    def test_context_manager(self):
        """PoseDetector3D should work as a context manager."""
        from pose3d.detector import PoseDetector3D

        with patch("pose3d.detector.PoseLandmarker") as MockLandmarker, \
             patch("pose3d.detector._get_model_path", return_value="/fake/model.task"):
            MockLandmarker.create_from_options.return_value = MagicMock()
            with PoseDetector3D() as det:
                assert det.model_complexity == 1

    def test_process_frame_with_detection(self):
        """process_frame should return a PoseResult when detection succeeds."""
        from pose3d.detector import PoseDetector3D

        frame = _blank_frame()
        mock_landmarker = _make_mock_landmarker(has_detection=True)

        with patch("pose3d.detector.PoseLandmarker") as MockLandmarker, \
             patch("pose3d.detector._get_model_path", return_value="/fake/model.task"), \
             patch("pose3d.detector.mp.Image") as MockImage:
            MockLandmarker.create_from_options.return_value = mock_landmarker
            MockImage.return_value = MagicMock()
            with PoseDetector3D(model_complexity=0) as det:
                result = det.process_frame(frame, frame_index=5)

        assert result is not None
        assert result.frame_index == 5
        assert result.keypoints_3d.shape == (33, 3)
        assert result.visibility.shape == (33,)

    def test_process_frame_without_detection(self):
        """process_frame should return None when no person is detected."""
        from pose3d.detector import PoseDetector3D

        frame = _blank_frame()
        mock_landmarker = _make_mock_landmarker(has_detection=False)

        with patch("pose3d.detector.PoseLandmarker") as MockLandmarker, \
             patch("pose3d.detector._get_model_path", return_value="/fake/model.task"), \
             patch("pose3d.detector.mp.Image") as MockImage:
            MockLandmarker.create_from_options.return_value = mock_landmarker
            MockImage.return_value = MagicMock()
            with PoseDetector3D(model_complexity=0) as det:
                result = det.process_frame(frame, frame_index=0)

        assert result is None

    def test_process_frames_list_length(self):
        """process_frames should return one result per input frame."""
        from pose3d.detector import PoseDetector3D

        frames = [_blank_frame() for _ in range(4)]
        mock_landmarker = _make_mock_landmarker(has_detection=False)

        with patch("pose3d.detector.PoseLandmarker") as MockLandmarker, \
             patch("pose3d.detector._get_model_path", return_value="/fake/model.task"), \
             patch("pose3d.detector.mp.Image") as MockImage:
            MockLandmarker.create_from_options.return_value = mock_landmarker
            MockImage.return_value = MagicMock()
            with PoseDetector3D(model_complexity=0) as det:
                results = det.process_frames(frames)

        assert len(results) == 4

    def test_pose_result_to_dict(self):
        """PoseResult.to_dict should return a JSON-serializable dict."""
        import json
        from pose3d.detector import PoseResult

        kp = _synthetic_keypoints()
        vis = _synthetic_visibility()
        r = PoseResult(frame_index=7, keypoints_3d=kp, visibility=vis)
        d = r.to_dict()
        assert d["frame_index"] == 7
        assert len(d["keypoints_3d"]) == 33
        # Ensure JSON-serialisable
        json.dumps(d)

    def test_pose_result_landmark_names(self):
        from pose3d.detector import PoseResult, LANDMARK_NAMES

        kp = _synthetic_keypoints()
        vis = _synthetic_visibility()
        r = PoseResult(frame_index=0, keypoints_3d=kp, visibility=vis)
        assert r.landmark_names == LANDMARK_NAMES
        assert len(r.landmark_names) == 33


# ---------------------------------------------------------------------------
# visualizer tests (no display required – Agg backend)
# ---------------------------------------------------------------------------

class TestVisualizer3D:
    def _make_result(self, frame_index: int = 0) -> "PoseResult":
        from pose3d.detector import PoseResult
        kp = _synthetic_keypoints()
        vis = _synthetic_visibility()
        return PoseResult(frame_index=frame_index, keypoints_3d=kp, visibility=vis)

    def test_plot_frame_creates_png(self, tmp_path):
        """plot_frame should write a PNG file."""
        from pose3d.visualizer import Visualizer3D

        result = self._make_result()
        out = str(tmp_path / "frame.png")
        viz = Visualizer3D()
        returned = viz.plot_frame(result, out)
        assert returned == out
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_animate_sequence_creates_gif(self, tmp_path):
        """animate_sequence should produce a GIF for a short result list."""
        from pose3d.visualizer import Visualizer3D

        results = [self._make_result(i) for i in range(3)]
        out = str(tmp_path / "anim.gif")
        viz = Visualizer3D()
        returned = viz.animate_sequence(results, out, fps=5)
        assert os.path.isfile(returned)
        assert os.path.getsize(returned) > 0

    def test_animate_sequence_skips_none(self, tmp_path):
        """None entries in the results list should be skipped."""
        from pose3d.visualizer import Visualizer3D

        results = [None, self._make_result(1), None, self._make_result(3)]
        out = str(tmp_path / "anim.gif")
        viz = Visualizer3D()
        returned = viz.animate_sequence(results, out, fps=5)
        assert os.path.isfile(returned)

    def test_animate_sequence_all_none_raises(self, tmp_path):
        from pose3d.visualizer import Visualizer3D

        out = str(tmp_path / "anim.gif")
        viz = Visualizer3D()
        with pytest.raises(ValueError):
            viz.animate_sequence([None, None], out, fps=5)

    def test_plot_trajectory(self, tmp_path):
        """plot_trajectory should create a PNG for selected landmarks."""
        from pose3d.visualizer import Visualizer3D

        results = [self._make_result(i) for i in range(5)]
        out = str(tmp_path / "traj.png")
        viz = Visualizer3D()
        returned = viz.plot_trajectory(results, [15, 16], out)
        assert os.path.isfile(returned)
        assert os.path.getsize(returned) > 0


# ---------------------------------------------------------------------------
# main() CLI tests
# ---------------------------------------------------------------------------

class TestMainCLI:
    def _patch_detector(self, has_detection: bool = False):
        """Return a context-manager stack that stubs the full PoseDetector3D."""
        from pose3d.detector import PoseResult

        def fake_process_frames(self_inner, frames):
            if not has_detection:
                return [None] * len(frames)
            kp = _synthetic_keypoints()
            vis = _synthetic_visibility()
            return [PoseResult(frame_index=i, keypoints_3d=kp, visibility=vis)
                    for i in range(len(frames))]

        mock_lm = _make_mock_landmarker(has_detection=has_detection)

        def fake_init(self_inner, model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5, model_path=None):
            self_inner.model_complexity = model_complexity
            self_inner._landmarker = mock_lm

        import contextlib
        @contextlib.contextmanager
        def _ctx():
            with patch("pose3d.detector.PoseDetector3D.__init__", fake_init), \
                 patch("pose3d.detector.PoseDetector3D.process_frames", fake_process_frames), \
                 patch("pose3d.detector.PoseDetector3D.close", lambda _: None):
                yield

        return _ctx()

    def test_missing_input_exits_nonzero(self):
        """CLI with a nonexistent input file should exit with non-zero code."""
        from main import main

        ret = main(["--input", "/nonexistent/video.mp4", "--output", "/tmp/out"])
        assert ret != 0

    def test_single_image_pipeline(self, tmp_path):
        """End-to-end: single blank image → output directory created."""
        import cv2
        from main import main

        img_path = str(tmp_path / "input.png")
        cv2.imwrite(img_path, _blank_frame())

        out_dir = str(tmp_path / "out")
        with self._patch_detector(has_detection=False):
            ret = main(["--input", img_path, "--output", out_dir,
                        "--model-complexity", "0"])
        # Return code 2 is acceptable (no person detected in blank frame)
        assert ret in (0, 2)
        assert os.path.isdir(out_dir)

    def test_image_sequence_pipeline(self, tmp_path):
        """End-to-end: directory of blank images → output directory created."""
        import cv2
        from main import main

        seq_dir = tmp_path / "seq"
        seq_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(seq_dir / f"frame_{i:03d}.png"), _blank_frame())

        out_dir = str(tmp_path / "out")
        with self._patch_detector(has_detection=False):
            ret = main(["--input", str(seq_dir), "--output", out_dir,
                        "--model-complexity", "0"])
        assert ret in (0, 2)
        assert os.path.isdir(out_dir)

    def test_json_export(self, tmp_path):
        """--save-json should write keypoints3d.json when poses are detected."""
        import cv2
        import json as json_mod
        from main import main

        img_path = str(tmp_path / "input.png")
        cv2.imwrite(img_path, _blank_frame())

        out_dir = str(tmp_path / "out")
        with self._patch_detector(has_detection=True):
            ret = main(["--input", img_path, "--output", out_dir,
                        "--model-complexity", "0", "--save-json"])
        assert ret == 0
        json_file = os.path.join(out_dir, "keypoints3d.json")
        assert os.path.isfile(json_file)
        data = json_mod.load(open(json_file))
        assert isinstance(data, list)
        assert data[0]["frame_index"] == 0
