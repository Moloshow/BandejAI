"""Ball tracking module using TrackNet.

This module provides the `BallTracker` class which maintains a centered 3-frame
rolling window [t-1, t, t+1] and delegates inference to the TrackNet PyTorch
architecture. Using a centered window eliminates the temporal lag that occurs with
a causal [t-2, t-1, t] window, at the cost of a 1-frame output delay. This is
acceptable in our offline batch pipeline.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from config import settings
from vision.ball_tracking.tracknet import TrackNet

logger = logging.getLogger(__name__)


class BallTracker:
    """Tracks the padel ball using a centered 3-frame sliding window TrackNet.

    The tracker buffers (preprocessed_frame, frame_idx) pairs. Once the buffer
    reaches 3 entries, it runs inference on the triplet and returns the ball
    position attributed to the MIDDLE frame (frame_idx at position 1 in the
    deque). This ensures zero temporal bias: the prediction for frame t uses
    equal context from t-1 and t+1.

    Because of the 1-frame lookahead, the tracker outputs a result that is
    always 1 frame behind the latest call to `update()`. The caller must handle
    this correctly (see `pipeline.py`).
    """

    TARGET_SIZE: tuple[int, int] = (640, 360)  # (W, H) - standard TrackNet input
    DETECTION_THRESHOLD: int = 128  # argmax value in [0, 255]; 128 == 50 % confidence

    def __init__(self, weights_path: Path | None = None) -> None:
        """Initialize the ball tracker and load model weights.

        Args:
            weights_path: Optional path to the .pt weights file.
                          Defaults to the path specified in config.py.
        """
        self.device = settings.torch_device

        # Each entry is (preprocessed_frame: np.ndarray, frame_idx: int)
        self._queue: deque[tuple[np.ndarray, int]] = deque(maxlen=3)

        self._model = TrackNet(in_channels=9, out_channels=256).to(self.device)
        self._model.eval()

        if weights_path is None:
            weights_path = settings.models_dir / settings.tracknet_weights

        if weights_path.exists():
            try:
                state_dict = torch.load(
                    weights_path, map_location=self.device, weights_only=False
                )
                self._model.load_state_dict(state_dict, strict=True)
                logger.info("BallTracker: loaded weights from %s", weights_path)
            except (RuntimeError, FileNotFoundError) as e:
                logger.error("BallTracker: failed to load weights: %s", e)
        else:
            logger.warning(
                "BallTracker: weights not found at %s. Predictions will be random.",
                weights_path,
            )

    def reset(self) -> None:
        """Clear the frame buffer. Call at the start of each rally."""
        self._queue.clear()

    def update(
        self, frame: np.ndarray, frame_idx: int
    ) -> tuple[tuple[int, int] | None, int | None]:
        """Process a new frame and return the ball position for the CURRENT frame.

        To fix prediction lag, we provide the frames to the model in reverse
        chronological order [t, t-1, t-2]. This forces the model to predict the
        position of the ball at the most recent frame (t), eliminating the
        delay observed when passing [t-2, t-1, t].

        Args:
            frame: BGR numpy array of the current video frame.
            frame_idx: The video frame index corresponding to `frame`.

        Returns:
            A tuple ``(ball_pos, target_frame_idx)`` where:
            - ``ball_pos`` is ``(x, y)`` in original-frame pixel coordinates,
              or ``None`` if the ball is not detected or the buffer is not full.
            - ``target_frame_idx`` is the current frame index (t),
              or ``None`` when ball_pos is ``None``.
        """
        preprocessed = self._preprocess(frame)
        self._queue.append((preprocessed, frame_idx))

        # We need all 3 slots filled before we can run inference
        if len(self._queue) < 3:
            return None, None

        # Reverse order: [t, t-1, t-2]. This makes the most recent frame the primary
        # target for the heatmap generator, fixing the temporal lag.
        frames = [self._queue[2][0], self._queue[1][0], self._queue[0][0]]
        target_frame_idx = self._queue[2][1]  # The newest frame (t)

        # Stack along channel axis: (H, W, 9)
        stacked = np.concatenate(frames, axis=2)
        tensor = (
            torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            output = self._model(tensor)

        # (1, 256, H, W) -> argmax over class dim -> (H, W) with values in [0, 255]
        heatmap = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.float32)

        _, max_val, _, max_loc = cv2.minMaxLoc(heatmap)

        if max_val < self.DETECTION_THRESHOLD:
            return None, None

        # Scale coordinates back to original frame resolution
        orig_h, orig_w = frame.shape[:2]
        x = int(max_loc[0] * orig_w / self.TARGET_SIZE[0])
        y = int(max_loc[1] * orig_h / self.TARGET_SIZE[1])

        return (x, y), target_frame_idx

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, convert to RGB, and normalize a BGR frame to [0, 1] float32.

        Args:
            frame: BGR numpy array.

        Returns:
            Normalized float32 array of shape (H, W, 3).
        """
        resized = cv2.resize(frame, self.TARGET_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0
