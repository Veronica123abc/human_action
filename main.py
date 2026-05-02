#!/usr/bin/env python3
"""
human_action – CLI entry point
================================
Generate 3-D representations of human actions from a video, image, or image sequence.

Examples
--------
Single image:
    python main.py --input photo.jpg --output output/

Video:
    python main.py --input walk.mp4 --output output/ --fps 25

Image sequence (directory):
    python main.py --input frames/ --output output/ --animate

"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pose3d.detector import PoseDetector3D
from pose3d.io_utils import load_frames, get_video_fps
from pose3d.visualizer import Visualizer3D


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D representation of human action.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a video file, image file, or directory of images.",
    )
    parser.add_argument(
        "--output", "-o", default="output",
        help="Directory to save results (default: output/).",
    )
    parser.add_argument(
        "--model-complexity", type=int, default=1, choices=[0, 1, 2],
        help=(
            "MediaPipe BlazePose model complexity: "
            "0=lite (~3 MB), 1=full (~7 MB, default), 2=heavy (~26 MB)."
        ),
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.5,
        help="Minimum confidence for person detection (default: 0.5).",
    )
    parser.add_argument(
        "--min-tracking-confidence", type=float, default=0.5,
        help="Minimum confidence for landmark tracking (default: 0.5).",
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0,
        help="Skip N frames between each processed frame (0 = process every frame).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Maximum number of frames to process (0 = all).",
    )
    parser.add_argument(
        "--fps", type=float, default=0,
        help="Frames per second for the output animation (0 = auto-detect from video).",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Save an animated GIF/MP4 of the 3-D skeleton sequence.",
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save 3-D keypoint data as JSON.",
    )
    parser.add_argument(
        "--trajectory-landmarks", nargs="+", type=int, default=[],
        metavar="IDX",
        help=(
            "Landmark indices whose 3-D trajectories should be plotted "
            "(e.g. --trajectory-landmarks 15 16 for wrists)."
        ),
    )
    parser.add_argument(
        "--vis-threshold", type=float, default=0.3,
        help="Minimum landmark visibility to render an edge (default: 0.3).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    os.makedirs(args.output, exist_ok=True)

    # ── load frames ───────────────────────────────────────────────────────────
    print(f"[INFO] Loading frames from: {args.input}")
    try:
        frames = load_frames(args.input, max_frames=args.max_frames, skip_frames=args.skip_frames)
    except (ValueError, IOError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    if not frames:
        print("[ERROR] No frames loaded. Check that the input path is valid.")
        return 1
    print(f"[INFO] {len(frames)} frame(s) loaded.")

    # ── 3D pose detection ─────────────────────────────────────────────────────
    model_names = {0: "lite (~3 MB)", 1: "full (~7 MB)", 2: "heavy (~26 MB)"}
    print(f"[INFO] Running 3D pose estimation "
          f"(MediaPipe BlazePose {model_names[args.model_complexity]}) …")

    with PoseDetector3D(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as detector:
        results = detector.process_frames(frames)

    n_detected = sum(1 for r in results if r is not None)
    print(f"[INFO] Pose detected in {n_detected}/{len(frames)} frame(s).")

    if n_detected == 0:
        print("[WARNING] No poses detected. "
              "Try lowering --min-detection-confidence or using a clearer input.")
        return 2

    # ── per-frame PNG plots ───────────────────────────────────────────────────
    viz = Visualizer3D(vis_threshold=args.vis_threshold)
    frames_dir = os.path.join(args.output, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    saved_pngs = []
    for result in results:
        if result is None:
            continue
        png_path = os.path.join(frames_dir, f"pose3d_{result.frame_index:05d}.png")
        viz.plot_frame(result, png_path)
        saved_pngs.append(png_path)
    print(f"[INFO] Saved {len(saved_pngs)} 3-D skeleton PNG(s) → {frames_dir}/")

    # ── animation ─────────────────────────────────────────────────────────────
    if args.animate or len(frames) > 1:
        fps = args.fps or get_video_fps(args.input) or 15.0
        anim_path = os.path.join(args.output, "animation.mp4")
        try:
            out = viz.animate_sequence(results, anim_path, fps=int(fps))
            print(f"[INFO] Animation saved → {out}")
        except Exception as exc:
            print(f"[WARNING] Animation failed ({exc}). Per-frame PNGs are still available.")

    # ── trajectory plot ───────────────────────────────────────────────────────
    if args.trajectory_landmarks:
        traj_path = os.path.join(args.output, "trajectory.png")
        viz.plot_trajectory(results, args.trajectory_landmarks, traj_path)
        print(f"[INFO] Trajectory plot saved → {traj_path}")

    # ── JSON export ───────────────────────────────────────────────────────────
    if args.save_json:
        json_path = os.path.join(args.output, "keypoints3d.json")
        data = [r.to_dict() for r in results if r is not None]
        with open(json_path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"[INFO] Keypoint data saved → {json_path}")

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
