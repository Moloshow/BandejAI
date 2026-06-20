"""Action Tube Extractor for Phase 4 (Action Recognition).

This module analyzes the ball and player trajectories to detect impact moments (hits),
and extracts a temporal window (Action Tube) around each hit for fine-grained 
stroke classification (Bandeja, Vibora, etc.).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ActionTubeExtractor:
    """Extracts Action Tubes (1.5s video clips around a hit) from tracking data."""

    def __init__(self, fps: float = 30.0, tube_duration_sec: float = 1.5):
        self.fps = fps
        self.tube_frames = int(tube_duration_sec * fps)
        self.pre_hit_frames = int(self.tube_frames * 0.3)  # 30% of tube is backswing
        self.post_hit_frames = self.tube_frames - self.pre_hit_frames

    def detect_hits(self, trajectories_path: Path | str) -> list[dict]:
        """Detect hit frames based on kinematic anomalies (acceleration spikes).

        Args:
            trajectories_path: Path to the rally_XXX_trajectories.json file.

        Returns:
            List of dictionaries containing hit metadata.
        """
        path = Path(trajectories_path)
        if not path.exists():
            logger.error(f"Trajectories file not found: {path}")
            return []

        with open(path) as f:
            data = json.load(f)

        ball_data = data.get("ball", {})
        players_data = data.get("players", {})

        if not ball_data:
            return []

        # Convert ball data to DataFrame for vector operations
        frames = sorted([int(k) for k in ball_data.keys()])
        df = pd.DataFrame(index=frames, columns=["x", "y"])
        for f_idx in frames:
            df.loc[f_idx] = ball_data[str(f_idx)]
            
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])

        # Calculate Velocity (v = dx/dt)
        df["vx"] = df["x"].diff()
        df["vy"] = df["y"].diff()
        
        # Calculate Acceleration (a = dv/dt)
        df["ax"] = df["vx"].diff()
        df["ay"] = df["vy"].diff()
        df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2)

        # Detect peaks in acceleration (hits, wall bounces, or ground bounces)
        # Lower threshold to 10.0 to catch softer shots like serves and returns
        acc_threshold = 10.0
        potential_hits = df[df["acc_mag"] > acc_threshold].index.tolist()

        valid_hits = []
        
        # Filter: A hit must happen NEAR a player
        for hit_frame in potential_hits:
            hit_pos = np.array([df.loc[hit_frame, "x"], df.loc[hit_frame, "y"]])
            
            # Find the closest player at this frame
            str_frame = str(hit_frame)
            if str_frame not in players_data:
                continue
                
            frame_players = players_data[str_frame]
            closest_slot = None
            min_score = float("inf")
            best_dist_to_box = float("inf")
            
            for slot, p_data in frame_players.items():
                # bbox is [x1, y1, x2, y2]
                bbox = p_data["bbox"]
                # Player center
                px = (bbox[0] + bbox[2]) / 2.0
                py = (bbox[1] + bbox[3]) / 2.0
                p_center = np.array([px, py])
                
                # Distance from ball to bounding box edges
                dx = max(bbox[0] - hit_pos[0], 0, hit_pos[0] - bbox[2])
                dy = max(bbox[1] - hit_pos[1], 0, hit_pos[1] - bbox[3])
                dist_to_box = math.hypot(dx, dy)
                
                # Distance to center (used as tie-breaker if ball is inside multiple bboxes)
                dist_to_center = np.linalg.norm(hit_pos - p_center)
                
                score = dist_to_box * 1000 + dist_to_center
                
                if score < min_score:
                    min_score = score
                    closest_slot = slot
                    best_dist_to_box = dist_to_box
            
            # If the ball is within ~100 pixels of the player's bounding box, it's a racket hit.
            if best_dist_to_box < 100.0:
                valid_hits.append({
                    "hit_frame": hit_frame,
                    "tube_start": max(0, hit_frame - self.pre_hit_frames),
                    "tube_end": hit_frame + self.post_hit_frames,
                    "player_slot": closest_slot,
                    "acceleration": round(float(df.loc[hit_frame, "acc_mag"]), 2),
                    "distance_to_player": round(best_dist_to_box, 2)
                })

        # Deduplicate hits that occur within 10 frames of each other (same kinetic event)
        dedup_hits = []
        for hit in valid_hits:
            if not dedup_hits or (hit["hit_frame"] - dedup_hits[-1]["hit_frame"] > 10):
                dedup_hits.append(hit)

        return dedup_hits

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = ActionTubeExtractor(fps=30.0, tube_duration_sec=1.5)
    hits = extractor.detect_hits("outputs/v17/rally_001_trajectories.json")
    for i, h in enumerate(hits):
        print(f"Stroke {i+1}: Frame {h['hit_frame']} by P{h['player_slot']} (Acc: {h['acceleration']} px/f2)")
