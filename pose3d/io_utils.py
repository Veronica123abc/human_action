"""
I/O utilities: load frames from a video file, a single image, or an image-sequence directory.
"""

from __future__ import annotations

import os
import re
from typing import Iterator, List

import cv2
import numpy as np

# Recognised image extensions
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# Recognised video extensions
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _IMAGE_EXTS


def _is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _VIDEO_EXTS


def load_frames(
    source: str,
    max_frames: int = 0,
    skip_frames: int = 0,
) -> List[np.ndarray]:
    """
    Load BGR frames from *source*.

    *source* can be:
      - A **video file** (e.g. ``walk.mp4``) → decoded frame-by-frame.
      - A **single image file** (e.g. ``pose.jpg``) → returned as one frame.
      - A **directory** containing image files → frames sorted by filename.

    Parameters
    ----------
    source : str
        Path to a video file, image file, or directory of images.
    max_frames : int
        Maximum number of frames to return (0 = unlimited).
    skip_frames : int
        Number of frames to skip between each loaded frame (0 = load every frame).
        E.g. ``skip_frames=1`` loads every other frame.

    Returns
    -------
    list of numpy.ndarray
        BGR frames (HxWx3, uint8).
    """
    if _is_image_file(source):
        return _load_single_image(source)
    if _is_video_file(source):
        return _load_video(source, max_frames, skip_frames)
    if os.path.isdir(source):
        return _load_image_sequence(source, max_frames, skip_frames)
    raise ValueError(
        f"Cannot determine input type for '{source}'. "
        "Provide a video file, image file, or directory of images."
    )


def iter_frames(
    source: str,
    skip_frames: int = 0,
) -> Iterator[np.ndarray]:
    """
    Generator version of :func:`load_frames` – streams frames one by one
    to avoid loading the entire video into memory.
    """
    if _is_image_file(source):
        img = cv2.imread(source)
        if img is None:
            raise IOError(f"Cannot read image: {source}")
        yield img
        return

    if _is_video_file(source):
        yield from _iter_video(source, skip_frames)
        return

    if os.path.isdir(source):
        image_paths = _sorted_image_paths(source)
        for idx, p in enumerate(image_paths):
            if skip_frames > 0 and idx % (skip_frames + 1) != 0:
                continue
            img = cv2.imread(p)
            if img is not None:
                yield img
        return

    raise ValueError(f"Cannot determine input type for '{source}'.")


# ── private helpers ───────────────────────────────────────────────────────────

def _load_single_image(path: str) -> List[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read image file: {path}")
    return [img]


def _load_video(path: str, max_frames: int, skip_frames: int) -> List[np.ndarray]:
    frames = []
    for frame in _iter_video(path, skip_frames):
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def _iter_video(path: str, skip_frames: int) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if skip_frames == 0 or idx % (skip_frames + 1) == 0:
                yield frame
            idx += 1
    finally:
        cap.release()


def _sorted_image_paths(directory: str) -> List[str]:
    """Return image paths in *directory*, sorted naturally by filename."""
    entries = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if _is_image_file(os.path.join(directory, f))
    ]

    def _natural_key(s: str):
        parts = re.split(r"(\d+)", os.path.basename(s))
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    return sorted(entries, key=_natural_key)


def _load_image_sequence(directory: str, max_frames: int, skip_frames: int) -> List[np.ndarray]:
    paths = _sorted_image_paths(directory)
    frames = []
    for idx, p in enumerate(paths):
        if skip_frames > 0 and idx % (skip_frames + 1) != 0:
            continue
        img = cv2.imread(p)
        if img is not None:
            frames.append(img)
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def get_video_fps(source: str) -> float:
    """Return the frame rate of *source* (1.0 for images / directories)."""
    if _is_video_file(source):
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps
    return 1.0
