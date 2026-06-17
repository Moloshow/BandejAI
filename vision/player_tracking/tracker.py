"""Player detection and tracking module using YOLOv8 and ByteTrack."""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = logging.getLogger(__name__)


class PlayerTracker:
    """Tracks players in a video using YOLOv8 and ByteTrack.

    Attributes:
        model (YOLO): The YOLOv8 model instance.
    """

    def __init__(self, model_path: str | Path = "yolov8n.pt") -> None:
        """Initialize the player tracker.

        Args:
            model_path: Path to the YOLOv8 model weights. Defaults to 'yolov8n.pt'
                       which will be downloaded automatically by ultralytics.
        """
        logger.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)

    def track_frame(
        self, frame: NDArray[np.uint8], persist: bool = True, conf: float = 0.10
    ) -> Results:
        """Track players in a single frame.

        Args:
            frame: The BGR image frame as a numpy array.
            persist: Whether to persist tracks across frames.

        Returns:
            The YOLO results object containing bounding boxes and track IDs.
        """
        # We only care about class 0 (person)
        # ultralytics returns a list of results, one per frame. We pass a single frame.
        results = self.model.track(
            frame,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=persist,
            conf=conf,
            imgsz=1088,
            verbose=False,
        )
        return results[0]

    def track_video(
        self, video_path: str | Path
    ) -> Generator[tuple[NDArray[np.uint8], Results], None, None]:
        """Track players across an entire video.

        Args:
            video_path: Path to the input video file.

        Yields:
            A tuple containing the original frame and the tracking Results.

        Raises:
            FileNotFoundError: If the video file cannot be opened.
        """
        path_str = str(video_path)
        cap = cv2.VideoCapture(path_str)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {path_str}")

        logger.info("Started tracking video: %s", path_str)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.track_frame(frame, persist=True)
                yield frame, results
        finally:
            cap.release()
            logger.info("Finished tracking video.")
