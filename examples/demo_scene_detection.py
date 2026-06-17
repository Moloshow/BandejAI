"""Demo script for extracting rallies from a match video."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.scene_detection import RallyExtractor

logger = logging.getLogger("bandejai.demo_scene_detection")


def main() -> None:
    """Entry point for the scene detection demo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Extract clean rallies from video")
    parser.add_argument(
        "--video", type=str, required=True, help="Path to full match video file"
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=5.0,
        help="Minimum rally length in seconds",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    extractor = RallyExtractor(min_scene_length_sec=args.min_length)
    rallies = extractor.extract_scenes(video_path)

    print("\n" + "=" * 40)
    print("RALLY SUMMARY:")
    for i, (start_s, end_s, start_f, end_f) in enumerate(rallies):
        duration = end_s - start_s
        print(
            f"Rally {i+1:02d}: {start_s:05.1f}s -> {end_s:05.1f}s "
            f"(Duration: {duration:.1f}s) | Frames: {start_f}->{end_f}"
        )
    print("=" * 40 + "\n")

    logger.info("You can now run main.py to process this video!")


if __name__ == "__main__":
    main()
