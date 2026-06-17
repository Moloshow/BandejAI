"""Scene detection and rally extraction module."""

from __future__ import annotations

import logging
from pathlib import Path

from scenedetect import ContentDetector, detect

logger = logging.getLogger(__name__)


class RallyExtractor:
    """Extracts continuous rallies from a full match video."""

    def __init__(
        self,
        min_scene_length_sec: float = 4.0,
        threshold: float = 27.0,
    ) -> None:
        """Initialize the RallyExtractor.

        Args:
            min_scene_length_sec: Minimum length of a scene in seconds to be considered a rally.
                                  Short clips are likely replays or transitions.
            threshold: The threshold for the ContentDetector (lower = more sensitive).
        """
        self.min_scene_length_sec = min_scene_length_sec
        self.threshold = threshold

    def extract_scenes(
        self, video_path: str | Path
    ) -> list[tuple[float, float, int, int]]:
        """Detect all scenes in a video and return the valid ones.

        Args:
            video_path: Path to the input video.

        Returns:
            A list of tuples containing (start_time_sec, end_time_sec, start_frame, end_frame)
            for each valid rally.
        """
        path_str = str(video_path)
        logger.info("Starting scene detection on %s", path_str)

        detector = ContentDetector(threshold=self.threshold)

        # detect returns a list of (FrameTimecode, FrameTimecode) representing start and end of scenes
        scene_list = detect(path_str, detector)

        logger.info("Found %d total scenes.", len(scene_list))

        valid_rallies = []

        for i, (start_timecode, end_timecode) in enumerate(scene_list):
            start_time = start_timecode.get_seconds()
            end_time = end_timecode.get_seconds()
            start_frame = start_timecode.get_frames()
            end_frame = end_timecode.get_frames()

            duration = end_time - start_time

            # Filter out scenes that are too short (replays, transitions)
            if duration >= self.min_scene_length_sec:
                valid_rallies.append((start_time, end_time, start_frame, end_frame))
            else:
                logger.debug("Scene %d rejected: too short (%.1fs)", i, duration)

        logger.info(
            "Found %d valid rallies (duration >= %.1fs).",
            len(valid_rallies),
            self.min_scene_length_sec,
        )

        return valid_rallies
