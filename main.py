"""BandejAI - Offline orchestrator (CLI).

This module is the single entry point for running the full analysis
pipeline on a recorded padel match video.

Usage:
    python main.py --video_path path/to/match.mp4 --output_dir results/

Pipeline phases (see README roadmap):
    1. Homography initialization (manual 12-point court keypoints)
    2. Player tracking (YOLOv8 + ByteTrack)
    3. Ball tracking (TrackNetV5)
    4. Pose estimation (ViTPose-L)
    5. Audio extraction & acoustic refereeing (Butterworth + YAMNet-256)
    6. Action recognition (PoseC3D + RGB Early-Fusion)
    7. Kinematics & tactical metrics (Kalman filter, "Corde")
    8. LLM coaching report generation (constrained decoding)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from config import settings
from core_math.homography.calibration_ui import HomographyUI
from core_math.homography.projector import CourtProjector
from vision.pipeline import process_player_tracking
from vision.scene_detection import RallyExtractor

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("bandejai")


# --------------------------------------------------------------------------- #
# CLI argument parsing
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace containing the parsed CLI arguments.

    """
    parser = argparse.ArgumentParser(
        prog="bandejai",
        description="BandejAI - Single-camera padel tactical analysis (offline)",
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        help="Path to the input match video file (.mp4, .avi, .mov).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config with pre-calibrated points and rallies.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=settings.output_dir,
        help=f"Directory for generated results (default: {settings.output_dir}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=settings.device,
        choices=["cuda", "cpu"],
        help=f"Compute device (default: {settings.device}).",
    )
    parser.add_argument(
        "--skip_audio",
        action="store_true",
        help="Skip acoustic analysis pipeline (useful for videos without audio).",
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Skip LLM coaching report generation (geometry-only output).",
    )
    parser.add_argument(
        "--show_video",
        action="store_true",
        help="Display the video with tracked trajectories in real-time.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Core Pipeline Phases
# --------------------------------------------------------------------------- #
# Global state to pass calibrated projector between phases
_projector: CourtProjector | None = None
_rallies: list[list[int]] = []


def run_homography_init(video_path: Path, config_data: dict | None = None) -> None:
    """Prompt the user for court keypoints and compute the homography matrix.

    Args:
        video_path: Path to the input video.
        config_data: Optional dict loaded from YAML config.
    """
    logger.info("[Phase 1] Homography initialization")
    global _projector, _rallies

    if config_data and "image_points" in config_data:
        import numpy as np
        logger.info("Loading homography calibration from config...")
        _projector = CourtProjector()
        image_points = np.array(config_data["image_points"], dtype=np.float64)
        points_mode = config_data.get("points_mode", 12)
        court_points = _projector.get_template(points_mode)
        _projector.compute_homography(image_points, court_points)

        if "rallies" in config_data:
            _rallies = config_data["rallies"]
            logger.info("Loaded %d rallies from config.", len(_rallies))
        else:
            logger.info("Extracting rallies using PySceneDetect...")
            extractor = RallyExtractor(min_scene_length_sec=5.0)
            scenes = extractor.extract_scenes(str(video_path))
            _rallies = [[s[2], s[3]] for s in scenes]
        return

    # 1. Extract rallies first so we pick a good frame
    logger.info("Extracting rallies using PySceneDetect...")
    extractor = RallyExtractor(min_scene_length_sec=5.0)
    scenes = extractor.extract_scenes(str(video_path))
    _rallies = [[s[2], s[3]] for s in scenes]

    if not _rallies:
        logger.warning("No rallies detected! Falling back to full video.")
        _rallies = [[0, float("inf")]]
        calib_frame = 0
    else:
        calib_frame = _rallies[0][0]

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, calib_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.error("Failed to read video for calibration.")
        sys.exit(1)

    ui = HomographyUI(frame, points_mode=12)
    ui.run(auto_close=True)

    if not ui.projector.is_calibrated:
        logger.error("Calibration failed or aborted.")
        sys.exit(1)

    _projector = ui.projector


def run_vision_pipeline(video_path: Path, output_dir: Path, show_video: bool) -> None:
    """Run player tracking pipeline.

    Args:
        video_path: Path to the input video file.
        output_dir: Output directory.
        show_video: Whether to display the video feed.
    """
    logger.info("[Phase 1-2] Vision pipeline (YOLO + Tracking + Smoothing)")
    global _projector, _rallies

    if not _projector:
        logger.error("Projector not calibrated! Cannot run vision pipeline.")
        sys.exit(1)

    process_player_tracking(
        video_path=video_path,
        rallies=_rallies,
        projector=_projector,
        output_dir=output_dir,
        show_window=show_video
    )


# --- Pending Pipeline Stubs (To be implemented) ---

def run_audio_pipeline(video_path: Path) -> None:
    """Extract audio and run acoustic refereeing (bounce detection + classification).

    Args:
        video_path: Path to the input video file.

    """
    logger.info("[Phase 3] Audio pipeline (Butterworth + YAMNet-256) - TODO")


def run_action_recognition() -> None:
    """Classify strokes (Bandeja, Vibora, Chiquita) via PoseC3D + RGB fusion."""
    logger.info("[Phase 4] Action recognition (PoseC3D + RGB) - TODO")


def run_kinematics_and_metrics() -> None:
    """Compute ball kinematics and tactical metrics (Corde, heatmaps)."""
    logger.info("[Phase 2-5] Kinematics & tactical metrics - TODO")


def run_llm_coach() -> None:
    """Generate the structured tactical report via local LLM with constrained decoding."""
    logger.info("[Phase 5] LLM Coach report generation - TODO")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    """Run the full BandejAI offline analysis pipeline.

    Returns:
        Exit code (0 for success, non-zero for failure).

    """
    args = parse_args()

    # --- Setup ---
    settings.device = args.device
    settings.apply_seed()
    logger.info(f"BandejAI starting - seed={settings.seed}, device={settings.torch_device}")
    # --- Config loading ---
    config_data = None
    if args.config:
        import yaml
        with open(args.config) as f:
            config_data = yaml.safe_load(f)

        if not args.video_path and "video_path" in config_data:
            args.video_path = Path(config_data["video_path"])

    if not args.video_path:
        logger.error("--video_path is required if not specified in config.")
        return 1

    logger.info(f"Input video: {args.video_path}")
    logger.info(f"Output dir:  {args.output_dir}")

    if not args.video_path.exists():
        logger.error(f"Video file not found: {args.video_path}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pipeline ---
    run_homography_init(args.video_path, config_data)
    run_vision_pipeline(args.video_path, args.output_dir, args.show_video)
    if not args.skip_audio:
        run_audio_pipeline(args.video_path)
    run_action_recognition()
    run_kinematics_and_metrics()
    if not args.skip_llm:
        run_llm_coach()

    logger.info("BandejAI analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
