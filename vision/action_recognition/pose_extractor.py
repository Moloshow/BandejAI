"""Pose Extractor for Phase 4 (Action Recognition).

This module extracts 17-DoF skeletal keypoints from player crops.

Design Philosophy:
    This module is designed to be a SWAPPABLE backend. The abstract base class
    `BasePoseExtractor` defines the interface. The current implementation uses
    YOLOv8-Pose for rapid prototyping. When MMPose is set up, the drop-in
    replacement is `ViTPoseLExtractor` - requiring ZERO changes to downstream code.

COCO Keypoint Format (17 DoF):
    0: nose       1: left_eye     2: right_eye    3: left_ear     4: right_ear
    5: left_shoulder  6: right_shoulder  7: left_elbow  8: right_elbow
    9: left_wrist  10: right_wrist  11: left_hip  12: right_hip
    13: left_knee  14: right_knee  15: left_ankle  16: right_ankle
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 13-DoF subset used for biomechanical analysis (drops face keypoints)
# Maps the 13-DoF index to the COCO-17 index
BIOMECH_13_DOF_MAP = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BIOMECH_13_DOF_NAMES = ["nose"] + COCO_KEYPOINT_NAMES[5:]


class SkeletonFrame(NamedTuple):
    """Container for a single frame's skeleton data."""
    frame_idx: int
    keypoints: np.ndarray   # Shape: (17, 2) -> (x, y) in original image coords
    confidence: np.ndarray  # Shape: (17,) -> confidence score per keypoint [0, 1]
    bbox: list[float]       # [x1, y1, x2, y2] of the player crop used


class BasePoseExtractor(ABC):
    """Abstract interface for all pose extractors (YOLO-Pose, ViTPose-L, etc.)."""

    @abstractmethod
    def extract(self, frame: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract keypoints from a single player crop.

        Args:
            frame: Full video frame (BGR).
            bbox: Bounding box [x1, y1, x2, y2] of the player in the full frame.

        Returns:
            Tuple of (keypoints [17x2], confidence [17]) or None if failed.
        """
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of the backend, used in logging and metadata."""
        ...


# ---------------------------------------------------------------------------
# Backend 1: YOLOv8-Pose (current, immediately available)
# ---------------------------------------------------------------------------
class YOLOPoseExtractor(BasePoseExtractor):
    """YOLO-Pose backend. Zero setup cost - uses existing ultralytics install.

    TODO (Phase 4 upgrade): Replace with ViTPoseLExtractor once MMPose is set up.
    """

    def __init__(self, model_name: str = "yolov8s-pose.pt", device: str = "cuda"):
        from ultralytics import YOLO
        logger.info("Loading YOLOv8-Pose model: %s on %s", model_name, device)
        self._model = YOLO(model_name)
        self._device = device
        logger.info("YOLOv8-Pose ready.")

    @property
    def backend_name(self) -> str:
        return "YOLOv8-Pose"

    def extract(self, frame: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract keypoints from a tight crop of the player.

        Top-Down approach: YOLO runs on the isolated player crop, not the full frame.
        This maximizes precision on the target subject, mimicking the ViTPose-L workflow.
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        # Add padding to ensure head/feet are not cropped
        pad = 20
        x1c = max(0, x1 - pad)
        y1c = max(0, y1 - pad)
        x2c = min(frame.shape[1], x2 + pad)
        y2c = min(frame.shape[0], y2 + pad)

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return None

        results = self._model(crop, device=self._device, verbose=False)

        if not results or results[0].keypoints is None:
            return None

        kp_data = results[0].keypoints
        if kp_data.xy is None or len(kp_data.xy) == 0:
            return None

        # Take the most confident person in the crop (should be only one)
        kp_xy = kp_data.xy[0].cpu().numpy()       # (17, 2) - coords in CROP space
        kp_conf = kp_data.conf[0].cpu().numpy()   # (17,) - confidence

        # Re-project keypoint coords from crop space back to FULL frame space
        kp_xy[:, 0] += x1c
        kp_xy[:, 1] += y1c

        return kp_xy, kp_conf


# ---------------------------------------------------------------------------
# Backend 2: ViTPose-L (FUTURE - drop-in replacement)
# ---------------------------------------------------------------------------
class ViTPoseLExtractor(BasePoseExtractor):
    """ViTPose-L backend via MMPose.

    This is the production-grade extractor for offline processing.
    Prerequisites: pip install mmcv mmdet mmpose (with correct CUDA build)

    Superiority over YOLO-Pose:
        - Global Attention Transformer sees through the padel fence mesh
        - Temporally stable (minimal jitter on wrist/elbow across frames)
        - CrowdPose pre-training improves robustness in multi-person occlusion
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        try:
            from mmpose.apis import init_model, inference_topdown
            self._init_model = init_model
            self._inference = inference_topdown
            self._model = init_model(config_path, checkpoint_path, device=device)
            logger.info("ViTPose-L loaded successfully.")
        except ImportError:
            raise ImportError(
                "MMPose not found. Install it with:\n"
                "pip install mmcv mmdet mmpose\n"
                "See: https://mmpose.readthedocs.io"
            )

    @property
    def backend_name(self) -> str:
        return "ViTPose-L"

    def extract(self, frame: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
        # MMPose top-down inference: pass a list of bounding boxes
        person = [{"bbox": bbox[:4], "bbox_score": 1.0}]
        results = self._inference(self._model, frame, person)
        if not results:
            return None
        kp = results[0].pred_instances
        return kp.keypoints[0], kp.keypoint_scores[0]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def build_pose_extractor(backend: str = "yolo", device: str = "cuda", **kwargs) -> BasePoseExtractor:
    """Factory to instantiate the correct backend.

    Args:
        backend: "yolo" for YOLOv8-Pose, "vitpose" for ViTPose-L.
        device: Compute device ("cuda" or "cpu").
        **kwargs: Additional arguments passed to the extractor constructor.

    Returns:
        A configured pose extractor ready for inference.
    """
    if backend == "yolo":
        return YOLOPoseExtractor(device=device, **kwargs)
    elif backend == "vitpose":
        return ViTPoseLExtractor(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown pose backend: '{backend}'. Choose 'yolo' or 'vitpose'.")
