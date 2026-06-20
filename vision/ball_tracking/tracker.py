"""Ball tracking module using TrackNet.

This module provides the `BallTracker` class which maintains a 3-frame rolling window
and delegates inference to the TrackNet PyTorch architecture.
"""

from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from config import settings
from vision.ball_tracking.tracknet import TrackNet


class BallTracker:
    """Tracks the padel ball using a 3-frame sliding window TrackNet."""

    def __init__(self, weights_path: Optional[Path] = None):
        """Initialize the ball tracker.

        Args:
            weights_path: Optional path to the .pt weights.
                          Defaults to the path specified in config.py.
        """
        self.device = settings.torch_device

        # The model takes 3 consecutive frames (9 channels)
        self.frame_queue = deque(maxlen=3)
        self.target_size = (640, 360)  # Standard TrackNet input size (W, H)

        # Initialize model with 256 output classes (TrackNet classic formulation)
        self.model = TrackNet(in_channels=9, out_channels=256).to(self.device)
        self.model.eval()

        if weights_path is None:
            weights_path = settings.models_dir / settings.tracknet_weights

        if weights_path.exists():
            try:
                # Load weights strictly to ensure exact architecture match
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                print(f"[BallTracker] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[BallTracker] Error loading weights: {e}")
        else:
            print(f"[BallTracker] Warning: Weights not found at {weights_path}. Inference will be random.")

    def reset(self) -> None:
        """Clear the frame buffer (useful when starting a new rally)."""
        self.frame_queue.clear()

    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Process a new frame and return the ball coordinate.

        Args:
            frame: BGR numpy array of the current video frame.

        Returns:
            Tuple of (x, y) coordinates of the ball, or None if the ball
            is not found or if the 3-frame buffer is not yet full.
        """
        # Resize to TrackNet resolution
        resized = cv2.resize(frame, self.target_size)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] and float32
        norm = rgb.astype(np.float32) / 255.0

        self.frame_queue.append(norm)

        # We need exactly 3 frames to predict the position at t
        if len(self.frame_queue) < 3:
            return None

        # Concatenate along the channel axis to get a (360, 640, 9) array
        # frames are [t-2, t-1, t]
        stacked = np.concatenate(self.frame_queue, axis=2)

        # Convert to PyTorch tensor format (B, C, H, W)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        # Output is (1, 256, 360, 640). Argmax over the class dimension to get the heatmap.
        heatmap_np = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Find the pixel with the maximum probability (intensity from 0 to 255)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_np.astype(np.float32))

        # Threshold (e.g. 128 / 255 -> 50% probability)
        if max_val < 128:
            return None

        # max_loc is (x, y) in the resized space (640, 360)
        # We need to project it back to the original frame resolution
        orig_h, orig_w = frame.shape[:2]
        ratio_x = orig_w / self.target_size[0]
        ratio_y = orig_h / self.target_size[1]

        final_x = int(max_loc[0] * ratio_x)
        final_y = int(max_loc[1] * ratio_y)

        return (final_x, final_y)
