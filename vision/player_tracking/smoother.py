"""Trajectory smoothing and interpolation for offline padel analytics."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TrajectorySmoother:
    """Fills gaps and smooths trajectories for player tracking data."""

    def __init__(self, max_gap_frames: int = 60) -> None:
        """Initialize smoother.

        Args:
            max_gap_frames: Maximum number of missing frames to interpolate.
                            If a player is lost for longer, we don't hallucinate.
        """
        self.max_gap_frames = max_gap_frames

    def process_rally(self, rally_data: list[dict]) -> dict[int, dict[int, dict]]:
        """Interpolate and smooth track data for a single rally.

        Args:
            rally_data: List of dicts, one per frame:
                        {'frame_idx': int, 'tracks': {slot_id: {'bbox': [x1, y1, x2, y2], 'court_pt': (cx, cy)}}}

        Returns:
            A nested dictionary mapping frame_idx -> slot_id -> {'bbox': ..., 'court_pt': ...}
            with the missing frames filled in.
        """
        if not rally_data:
            return {}

        # Flatten the data to load into Pandas
        rows = []
        for frame_data in rally_data:
            f_idx = frame_data["frame_idx"]
            for slot, track_info in frame_data["tracks"].items():
                rows.append({
                    "frame": f_idx,
                    "slot": slot,
                    "x1": track_info["bbox"][0],
                    "y1": track_info["bbox"][1],
                    "x2": track_info["bbox"][2],
                    "y2": track_info["bbox"][3],
                    "cx": track_info["court_pt"][0],
                    "cy": track_info["court_pt"][1],
                })

        if not rows:
            return {}

        df = pd.DataFrame(rows)

        # We need a complete index of all frames from min to max, for each slot
        min_f = df["frame"].min()
        max_f = df["frame"].max()
        all_frames = pd.RangeIndex(min_f, max_f + 1, name="frame")

        # Create a MultiIndex of (frame, slot) covering all possible combinations
        slots = df["slot"].unique()
        multi_idx = pd.MultiIndex.from_product([all_frames, slots], names=["frame", "slot"])

        df = df.set_index(["frame", "slot"]).reindex(multi_idx).reset_index()

        # Interpolate missing values per slot
        df = df.sort_values(["slot", "frame"])

        # We only interpolate if the gap is <= max_gap_frames
        interpolated_cols = ["x1", "y1", "x2", "y2", "cx", "cy"]
        df[interpolated_cols] = df.groupby("slot")[interpolated_cols].transform(
            lambda x: x.interpolate(method="linear", limit=self.max_gap_frames, limit_direction="both")
        )

        # Smooth the trajectories using a rolling window
        df[interpolated_cols] = df.groupby("slot")[interpolated_cols].transform(
            lambda x: x.rolling(window=5, min_periods=1, center=True).mean()
        )

        # Drop rows that are still NaN (meaning they were gaps larger than max_gap_frames or start/end bounds)
        df = df.dropna(subset=["cx"])

        # Reconstruct the output dictionary
        smoothed_data = {}
        for _, row in df.iterrows():
            f_idx = int(row["frame"])
            slot = int(row["slot"])

            if f_idx not in smoothed_data:
                smoothed_data[f_idx] = {}

            smoothed_data[f_idx][slot] = {
                "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
                "court_pt": (row["cx"], row["cy"])
            }

        return smoothed_data
