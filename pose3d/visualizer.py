"""
3D visualisation utilities for human pose.

Provides:
  - ``Visualizer3D.plot_frame``         – single-frame 3-D skeleton plot saved to PNG
  - ``Visualizer3D.animate_sequence``   – multi-frame animation saved to MP4 / GIF
  - ``Visualizer3D.plot_trajectory``    – joint trajectory over time
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe in all environments
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers the projection
import numpy as np

from pose3d.detector import PoseResult, SKELETON_EDGES

# Colour map: left body = blue, right body = red, centre = green
_LEFT_LANDMARKS  = {1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_LANDMARKS = {4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

_EDGE_COLOR_LEFT   = "#4A90D9"   # blue
_EDGE_COLOR_RIGHT  = "#E74C3C"   # red
_EDGE_COLOR_CENTER = "#2ECC71"   # green


def _edge_color(i: int, j: int) -> str:
    if i in _LEFT_LANDMARKS and j in _LEFT_LANDMARKS:
        return _EDGE_COLOR_LEFT
    if i in _RIGHT_LANDMARKS and j in _RIGHT_LANDMARKS:
        return _EDGE_COLOR_RIGHT
    return _EDGE_COLOR_CENTER


def _setup_axes(ax: plt.Axes, title: str = "") -> None:
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")  # depth shown on Y-axis for intuitive view
    ax.set_zlabel("Y (m)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)


def _draw_skeleton(
    ax: plt.Axes,
    keypoints: np.ndarray,
    visibility: np.ndarray,
    vis_threshold: float = 0.3,
) -> None:
    """Draw skeleton edges and landmark dots on *ax*."""
    # Remap: world coords are (x, y, z) where y points down.
    # We swap y/z so the figure has z pointing up (natural upright pose).
    xs = keypoints[:, 0]
    ys = keypoints[:, 2]   # depth
    zs = -keypoints[:, 1]  # negate y so up is up

    for i, j in SKELETON_EDGES:
        if visibility[i] < vis_threshold or visibility[j] < vis_threshold:
            continue
        ax.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            [zs[i], zs[j]],
            color=_edge_color(i, j),
            linewidth=2,
            alpha=0.85,
        )

    # Draw joints
    visible_mask = visibility >= vis_threshold
    ax.scatter(
        xs[visible_mask], ys[visible_mask], zs[visible_mask],
        color="white", edgecolors="black", s=20, zorder=5,
    )


class Visualizer3D:
    """
    Creates 3-D visualisations from ``PoseResult`` objects.

    Parameters
    ----------
    figsize : tuple
        Figure size in inches (width, height).
    dpi : int
        Dots per inch for saved figures.
    vis_threshold : float
        Minimum landmark visibility to include an edge/joint in the plot.
    """

    def __init__(
        self,
        figsize: tuple = (8, 8),
        dpi: int = 100,
        vis_threshold: float = 0.3,
    ) -> None:
        self.figsize = figsize
        self.dpi = dpi
        self.vis_threshold = vis_threshold

    # ── single frame ─────────────────────────────────────────────────────────

    def plot_frame(
        self,
        result: PoseResult,
        output_path: str,
        title: str = "",
    ) -> str:
        """
        Save a 3-D skeleton plot for one frame to *output_path* (PNG).

        Returns the output path.
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        _setup_axes(ax, title or f"Frame {result.frame_index}")
        _draw_skeleton(ax, result.keypoints_3d, result.visibility, self.vis_threshold)
        self._set_equal_aspect(ax, result.keypoints_3d)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    # ── animation ────────────────────────────────────────────────────────────

    def animate_sequence(
        self,
        results: List[Optional[PoseResult]],
        output_path: str,
        fps: int = 15,
        title: str = "",
    ) -> str:
        """
        Save an animation of the pose sequence to *output_path*.

        Supports MP4 (requires ffmpeg) and GIF (always available).
        Returns the output path.
        """
        valid = [r for r in results if r is not None]
        if not valid:
            raise ValueError("No valid pose results to animate.")

        # Compute global axis limits across all frames
        all_kp = np.concatenate([r.keypoints_3d for r in valid], axis=0)
        xs, ys, zs = all_kp[:, 0], all_kp[:, 2], -all_kp[:, 1]
        margin = 0.15
        xlim = (xs.min() - margin, xs.max() + margin)
        ylim = (ys.min() - margin, ys.max() + margin)
        zlim = (zs.min() - margin, zs.max() + margin)

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        _setup_axes(ax, title)

        def _update(frame_idx: int):
            ax.cla()
            _setup_axes(ax, f"{title}  frame {valid[frame_idx].frame_index}")
            _draw_skeleton(
                ax,
                valid[frame_idx].keypoints_3d,
                valid[frame_idx].visibility,
                self.vis_threshold,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

        ani = animation.FuncAnimation(
            fig, _update, frames=len(valid), interval=1000 // fps, blit=False
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
            ani.save(output_path, writer=writer)
        else:
            try:
                ani.save(output_path, writer="ffmpeg", fps=fps)
            except Exception:
                # Fallback to GIF if ffmpeg is unavailable
                gif_path = os.path.splitext(output_path)[0] + ".gif"
                writer = animation.PillowWriter(fps=fps)
                ani.save(gif_path, writer=writer)
                output_path = gif_path
        plt.close(fig)
        return output_path

    # ── trajectory plot ───────────────────────────────────────────────────────

    def plot_trajectory(
        self,
        results: List[Optional[PoseResult]],
        landmark_indices: List[int],
        output_path: str,
        title: str = "",
    ) -> str:
        """
        Plot the 3-D trajectory of selected landmarks over time.

        Parameters
        ----------
        landmark_indices : list[int]
            Indices into the 33-landmark array (e.g. [15, 16] for wrists).
        """
        from pose3d.detector import LANDMARK_NAMES
        valid = [r for r in results if r is not None]
        if not valid:
            raise ValueError("No valid pose results.")

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")
        _setup_axes(ax, title or "Joint Trajectories")

        cmap = plt.get_cmap("tab10")
        for c_idx, lm_idx in enumerate(landmark_indices):
            coords = np.array([r.keypoints_3d[lm_idx] for r in valid])
            xs = coords[:, 0]
            ys = coords[:, 2]
            zs = -coords[:, 1]
            name = LANDMARK_NAMES[lm_idx] if lm_idx < len(LANDMARK_NAMES) else str(lm_idx)
            color = cmap(c_idx % 10)
            ax.plot(xs, ys, zs, color=color, linewidth=2, label=name)
            ax.scatter(xs[0], ys[0], zs[0], color=color, marker="o", s=60)
            ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker="^", s=60)

        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _set_equal_aspect(ax: plt.Axes, keypoints: np.ndarray) -> None:
        """Set equal aspect ratio so the skeleton is not distorted."""
        xs = keypoints[:, 0]
        ys = keypoints[:, 2]
        zs = -keypoints[:, 1]
        max_range = max(
            xs.max() - xs.min(),
            ys.max() - ys.min(),
            zs.max() - zs.min(),
        ) / 2.0 or 0.5
        mid_x = (xs.max() + xs.min()) / 2
        mid_y = (ys.max() + ys.min()) / 2
        mid_z = (zs.max() + zs.min()) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
