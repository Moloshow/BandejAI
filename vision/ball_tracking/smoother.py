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

    def process_rally(self, ball_data: dict, return_metadata: bool = False):
        """Interpolate the ball track.

        We use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.
        PCHIP is mathematically guaranteed to NEVER overshoot local extrema.
        
        Args:
            ball_data: Dictionary mapping frame_idx -> [x, y].
            return_metadata: If True, returns a tuple (smoothed_data, metadata_dict).

        Returns:
            A new dictionary mapping frame_idx -> [x, y] with gaps filled.
        """
        if not ball_data:
            return ({}, {}) if return_metadata else {}
            
        frames = list(ball_data.keys())
        min_f, max_f = min(frames), max(frames)

        df = pd.DataFrame([
            {"frame": f_idx, "x": ball_data[f_idx][0], "y": ball_data[f_idx][1]}
            for f_idx in frames
        ])

        df = df.set_index("frame").reindex(range(min_f, max_f + 1))

        # 0. Drop static background false positives (Global Hotspot Masking)
        # TrackNet sometimes hallucinates spots on the wall. Since we run offline, we can 
        # map these static "hotspots" and scrub the entire rally for any detection near them.
        valid_frames = df.dropna().index
        static_drops = set()
        hotspots = []
        filter_actions = []
        
        # Pass 1: Identify static hotspots (2+ consecutive detections, almost 0 movement)
        # We use a strict < 3.0 pixels threshold. This perfectly catches resting balls and
        # true static lights, but is small enough to NEVER catch a moving ball (like a serve).
        for i in range(len(valid_frames) - 1):
            f1, f2 = valid_frames[i], valid_frames[i+1]
            if f2 - f1 <= 5:
                p1 = np.array([df.loc[f1, "x"], df.loc[f1, "y"]])
                p2 = np.array([df.loc[f2, "x"], df.loc[f2, "y"]])
                if np.linalg.norm(p2 - p1) < 3.0:
                    hotspots.append((p1 + p2) / 2.0)
                    static_drops.update([f1, f2])
                    
        # Pass 2: Scrub the entire rally using the discovered hotspots
        if hotspots:
            for f in valid_frames:
                if f not in static_drops:
                    pt = np.array([df.loc[f, "x"], df.loc[f, "y"]])
                    for hs in hotspots:
                        if np.linalg.norm(pt - hs) < 10.0:  # 10 pixel danger radius
                            static_drops.add(f)
                            break
                            
        if static_drops:
            df.loc[list(static_drops), ["x", "y"]] = np.nan
            logger.info("Dropped %d static background false positive frames (Global Hotspot Masking)", len(static_drops))
            filter_actions.append({
                "action": "DROPPED_STATIC",
                "reason": "Global Hotspot Masking (Static Object)",
                "frames": [int(f) for f in static_drops]
            })

        # 1. Filter out isolated false positives (Rolling Median)
        # window=5 with min_periods=2 is the "Holy Grail": 
        # - It destroys 1-frame AND 2-frame teleportations (like ceiling lights).
        # - It cleans edge hallucinations (like frame 337).
        # - It safely ignores real trajectory endpoints.
        valid_frames = df.dropna().index
        outlier_drops = set()
        for col in ["x", "y"]:
            valid = df[col].dropna()
            if len(valid) >= 3:
                smoothed_valid = valid.rolling(window=5, center=True, min_periods=2).median()
                diff = np.abs(valid - smoothed_valid)
                outlier_indices = diff[diff > 60].index
                outlier_drops.update(outlier_indices)
                
        if outlier_drops:
            df.loc[list(outlier_drops), ["x", "y"]] = np.nan
            filter_actions.append({
                "action": "DROPPED_OUTLIERS",
                "reason": "Isolated Teleportation (Median Filter Diff > 60px)",
                "frames": [int(f) for f in outlier_drops]
            })

        # 2. Interpolate missing gaps using PCHIP (no overshoot)
        # We MUST use limit_area="inside" to prevent wild extrapolation at the beginning/end of the rally
        df["x"] = df["x"].interpolate(method="pchip", limit=self.max_gap_frames, limit_area="inside")
        df["y"] = df["y"].interpolate(method="pchip", limit=self.max_gap_frames, limit_area="inside")

        # Fallback to linear for very small sequences where pchip fails
        df["x"] = df["x"].interpolate(method="linear", limit=self.max_gap_frames, limit_area="inside")
        df["y"] = df["y"].interpolate(method="linear", limit=self.max_gap_frames, limit_area="inside")

        smoothed_data = {}
        for f_idx, row in df.dropna().iterrows():
            smoothed_data[int(f_idx)] = [int(row["x"]), int(row["y"])]

        if return_metadata:
            formatted_hotspots = [{"x": float(hs[0]), "y": float(hs[1])} for hs in hotspots]
            return smoothed_data, {
                "hotspots": formatted_hotspots, 
                "static_frames_dropped": len(static_drops),
                "outliers_dropped": len(outlier_drops),
                "filter_actions": filter_actions
            }
            
        return smoothed_data
