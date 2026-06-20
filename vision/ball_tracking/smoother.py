"""Ball trajectory smoothing and interpolation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BallSmoother:
    """Fills gaps and smooths the ball trajectory robustly."""

    def __init__(self, max_gap_frames: int = 15) -> None:
        """Initialize the ball smoother.

        Args:
            max_gap_frames: Maximum number of consecutive missing frames to interpolate.
        """
        self.max_gap_frames = max_gap_frames

    def process_rally(self, ball_data: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
        """Interpolate the ball track.

        We use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.
        PCHIP is mathematically guaranteed to NEVER overshoot local extrema.
        - On a straight line (smash), it stays perfectly straight.
        - On a curve (lob), it connects smoothly without bulging.
        - It completely eliminates the "louche" effect and the wild oscillations 
          caused by standard polynomials or Kalman filters.

        Args:
            ball_data: Dictionary mapping frame_idx -> (x, y).

        Returns:
            A new dictionary mapping frame_idx -> (x, y) with gaps filled.
        """
        if not ball_data:
            return {}

        frames = list(ball_data.keys())
        min_f, max_f = min(frames), max(frames)

        df = pd.DataFrame([
            {"frame": f_idx, "x": ball_data[f_idx][0], "y": ball_data[f_idx][1]}
            for f_idx in frames
        ])

        df = df.set_index("frame").reindex(range(min_f, max_f + 1))

        # 0. Drop static background false positives (e.g., TrackNet locking onto a white spot on the wall)
        # A real ball in play NEVER stays within a tiny pixel radius across 2 consecutive detections.
        valid_frames = df.dropna().index
        static_drops = set()
        for i in range(len(valid_frames) - 1):
            f1, f2 = valid_frames[i], valid_frames[i+1]
            # If these 2 detections happen within a short time window (e.g., 5 frames)
            if f2 - f1 <= 5:
                p1 = np.array([df.loc[f1, "x"], df.loc[f1, "y"]])
                p2 = np.array([df.loc[f2, "x"], df.loc[f2, "y"]])
                
                # If the distance between these points is tiny, it's a static object!
                if np.linalg.norm(p2 - p1) < 3.0:
                    static_drops.update([f1, f2])
                    
        if static_drops:
            df.loc[list(static_drops), ["x", "y"]] = np.nan
            logger.info("Dropped %d static background false positive frames", len(static_drops))

        # 1. Filter out isolated false positives before interpolating
        for col in ["x", "y"]:
            valid = df[col].dropna()
            if len(valid) >= 3:
                smoothed_valid = valid.rolling(window=3, center=True, min_periods=1).median()
                diff = np.abs(valid - smoothed_valid)
                outlier_indices = diff[diff > 50].index
                df.loc[outlier_indices, col] = np.nan

        # 2. Interpolate missing gaps using PCHIP (no overshoot)
        df["x"] = df["x"].interpolate(method="pchip", limit=self.max_gap_frames, limit_direction="both")
        df["y"] = df["y"].interpolate(method="pchip", limit=self.max_gap_frames, limit_direction="both")

        # Fallback to linear for very small sequences where pchip fails
        df["x"] = df["x"].interpolate(method="linear", limit=self.max_gap_frames, limit_direction="both")
        df["y"] = df["y"].interpolate(method="linear", limit=self.max_gap_frames, limit_direction="both")

        smoothed_data = {}
        for f_idx, row in df.dropna().iterrows():
            smoothed_data[int(f_idx)] = (int(round(row["x"])), int(round(row["y"])))

        return smoothed_data
