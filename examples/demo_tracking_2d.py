"""Interactive demo combining player tracking and 2D court projection.

Usage:
    python examples/demo_tracking_2d.py --video data/sample_match.mp4.webm --points 12
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_math.homography.projector import CourtProjector  # noqa: E402
from examples.demo_homography import HomographyDemo  # noqa: E402
from vision.player_tracking import PlayerTracker  # noqa: E402

logger = logging.getLogger("bandejai.demo_tracking")


def create_birds_eye_view(
    projector: CourtProjector, scale: int = 30, margin: int = 40
) -> NDArray[np.uint8]:
    """Create a blank 2D bird's-eye view of the court."""
    w_px = int(projector.COURT_WIDTH_M * scale)
    l_px = int(projector.COURT_LENGTH_M * scale)

    court_img = np.ones((l_px + 2 * margin, w_px + 2 * margin, 3), dtype=np.uint8) * 30

    def court_to_img(x_m: float, y_m: float) -> tuple[int, int]:
        # Invert Y so near baseline (y=0) is at the bottom of the minimap
        return int(margin + x_m * scale), int(margin + (projector.COURT_LENGTH_M - y_m) * scale)

    # Draw court background (green)
    p1 = court_to_img(0, 0)
    p2 = court_to_img(projector.COURT_WIDTH_M, projector.COURT_LENGTH_M)
    cv2.rectangle(court_img, p1, p2, (34, 89, 56), -1)

    # Draw court lines
    w = projector.COURT_WIDTH_M
    length = projector.COURT_LENGTH_M
    s = projector.SERVICE_LINE_DIST_M

    court_lines = [
        ([0, 0], [w, 0], (255, 255, 255)),
        ([0, length], [w, length], (255, 255, 255)),
        ([0, 0], [0, length], (255, 255, 255)),
        ([w, 0], [w, length], (255, 255, 255)),
        ([0, s], [w, s], (255, 255, 255)),
        ([0, length - s], [w, length - s], (255, 255, 255)),
        ([w / 2, s], [w / 2, length / 2], (255, 255, 255)),
        ([w / 2, length / 2], [w / 2, length - s], (255, 255, 255)),
    ]
    for p1_c, p2_c, color in court_lines:
        cv2.line(court_img, court_to_img(*p1_c), court_to_img(*p2_c), color, 1)

    # Net
    cv2.line(
        court_img,
        court_to_img(0, length / 2),
        court_to_img(w, length / 2),
        (0, 255, 255),
        2,
    )

    return court_img


def main() -> None:
    """Entry point for the tracking demo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="2D Player Tracking Demo")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--config", type=str, help="Path to YAML config with pre-calibrated points")
    parser.add_argument(
        "--points",
        type=int,
        choices=[15, 12, 10, 6, 4],
        default=12,
        help="Number of keypoints for homography calibration",
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Frame index to start video and calibration"
    )
    args = parser.parse_args()

    projector = CourtProjector()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        args.video = cfg.get("video_path", args.video)
        args.points = cfg.get("points_mode", args.points)
        image_points = np.array(cfg["image_points"], dtype=np.float64)

        if not args.video:
            logger.error("No video_path found in config and --video not provided.")
            sys.exit(1)

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", args.video)
            sys.exit(1)

        if args.frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            logger.error("Cannot read video frame.")
            sys.exit(1)

        court_points = projector.get_template(args.points)
        projector.compute_homography(image_points, court_points)
        logger.info("Loaded homography calibration from %s", args.config)

    else:
        if not args.video:
            logger.error("--video is required if --config is not provided.")
            parser.print_help()
            sys.exit(1)

        # 1. Read the specified frame for calibration
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", args.video)
            sys.exit(1)

        if args.frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            logger.error("Cannot read first frame of video.")
            sys.exit(1)

        # 2. Run Homography Calibration
        logger.info("Starting homography calibration...")
        calibration_demo = HomographyDemo(first_frame, points_mode=args.points)
        calibration_demo.run(auto_close=True)

        if not calibration_demo.projector.is_calibrated:
            logger.error("Homography calibration was not completed. Exiting.")
            sys.exit(1)

        projector = calibration_demo.projector

    # 3. Initialize Tracker
    logger.info("Initializing PlayerTracker...")
    tracker = PlayerTracker("yolov8n.pt")

    # 4. Process Video
    window_main = "Player Tracking"
    window_2d = "2D Mini-Map"
    cv2.namedWindow(window_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_2d, cv2.WINDOW_AUTOSIZE)

    # Resize main window to a reasonable height (e.g., 700px) and preserve aspect ratio
    if "first_frame" in locals() and first_frame is not None:
        h, w = first_frame.shape[:2]
        target_h = 700
        target_w = int(w * (target_h / h))
        cv2.resizeWindow(window_main, target_w, target_h)
        # Move windows side by side
        cv2.moveWindow(window_main, 50, 50)
        cv2.moveWindow(window_2d, 50 + target_w + 20, 50)

    scale = 30
    margin = 40

    def court_to_img(x_m: float, y_m: float) -> tuple[int, int]:
        return int(margin + x_m * scale), int(margin + (projector.COURT_LENGTH_M - y_m) * scale)

    # Define some colors for different track IDs
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track players
        results = tracker.track_frame(frame, persist=True)

        # Base 2D map
        minimap = create_birds_eye_view(projector, scale, margin)

        # Draw results
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids, strict=False):
                x1, y1, x2, y2 = box
                color = colors[track_id % len(colors)]

                # Draw on main frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Draw bottom center on main frame
                bc_x, bc_y = int((x1 + x2) / 2), int(y2)
                cv2.circle(frame, (bc_x, bc_y), 4, (0, 0, 255), -1)

                # Project to 2D
                try:
                    court_pt = projector.project_bounding_box_bottom(box)
                    # Draw on minimap
                    mm_x, mm_y = court_to_img(court_pt[0], court_pt[1])
                    cv2.circle(minimap, (mm_x, mm_y), 8, color, -1)
                    cv2.putText(
                        minimap,
                        str(track_id),
                        (mm_x - 5, mm_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                except RuntimeError:
                    pass

        cv2.imshow(window_main, frame)
        cv2.imshow(window_2d, minimap)

        # 30 ms delay, q to quit
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
