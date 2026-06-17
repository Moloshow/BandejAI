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
        required=True,
        help="Path to the input match video file (.mp4, .avi, .mov).",
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
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Pipeline stubs (Phase 0 - to be implemented in subsequent phases)
# --------------------------------------------------------------------------- #
def run_homography_init(video_path: Path) -> None:
    """Prompt the user for 12 court keypoints and compute the homography matrix.

    Args:
        video_path: Path to the input video (used to extract the first frame).

    """
    logger.info("[Phase 1] Homography initialization - TODO")


def run_vision_pipeline(video_path: Path) -> None:
    """Run player tracking, ball tracking, and pose estimation.

    Args:
        video_path: Path to the input video file.

    """
    logger.info("[Phase 1-2] Vision pipeline (YOLO + TrackNet + ViTPose) - TODO")


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
    logger.info(f"Input video: {args.video_path}")
    logger.info(f"Output dir:  {args.output_dir}")

    if not args.video_path.exists():
        logger.error(f"Video file not found: {args.video_path}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pipeline ---
    run_homography_init(args.video_path)
    run_vision_pipeline(args.video_path)
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
