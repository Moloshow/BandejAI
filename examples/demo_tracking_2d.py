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
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_math.homography.projector import CourtProjector  # noqa: E402
from examples.demo_homography import HomographyDemo  # noqa: E402
from vision.player_tracking import PlayerMerger, PlayerTracker, TrajectorySmoother  # noqa: E402

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
    parser.add_argument(
        "--output", type=str, help="Directory to save output videos (optional)"
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
        rallies = cfg.get("rallies", [])

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

        logger.info("No config provided. Running Scene Detection to find a perfect calibration frame...")
        rallies = []
        try:
            from vision.scene_detection import RallyExtractor
            extractor = RallyExtractor(min_scene_length_sec=5.0)
            scenes = extractor.extract_scenes(args.video)
            rallies = [[s[2], s[3]] for s in scenes]  # start_frame, end_frame
        except ImportError:
            logger.warning("scenedetect not installed. Pip install scenedetect to enable auto-rally extraction.")

        if not rallies:
            logger.warning("No rallies detected! Falling back to full video.")
            rallies = [[0, float("inf")]]

        # 1. Read the specified frame for calibration
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", args.video)
            sys.exit(1)

        # Use args.frame if provided, otherwise use the start of the first rally!
        calib_frame = args.frame if args.frame > 0 else rallies[0][0]
        if calib_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, calib_frame)

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            logger.error("Cannot read first frame of video.")
            sys.exit(1)

        # 2. Run Homography Calibration
        logger.info("Starting homography calibration on frame %d...", calib_frame)
        calibration_demo = HomographyDemo(first_frame, points_mode=args.points)
        calibration_demo.run(auto_close=True)

        if not calibration_demo.projector.is_calibrated:
            logger.error("Homography calibration was not completed. Exiting.")
            sys.exit(1)

        projector = calibration_demo.projector

    # 4. Initialize Tracker
    logger.info("Initializing PlayerTracker...")
    tracker = PlayerTracker("yolov8n.pt")
    smoother = TrajectorySmoother(max_gap_frames=60)

    # 4. Process Video
    window_main = "Padelytics Tracking Dashboard"
    cv2.namedWindow(window_main, cv2.WINDOW_AUTOSIZE)

    scale = 30
    margin = 40

    def court_to_img(x_m: float, y_m: float) -> tuple[int, int]:
        return int(margin + x_m * scale), int(margin + (projector.COURT_LENGTH_M - y_m) * scale)

    # Loop over all rallies
    for i, (start_f, end_f) in enumerate(rallies):
        logger.info("=== Processing Rally %d/%d (Frames %d -> %d) ===", i+1, len(rallies), start_f, end_f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        # Reset merger for a clean state per rally
        merger = PlayerMerger(projector, max_lost_frames=60)
        user_quit = False

        rally_data = []
        valid_detections = 0

        # --- PASS 1: TRACKING ---
        total_frames = end_f - start_f
        pbar = tqdm(total=total_frames, desc=f"Tracking Rally {i+1}")

        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_f:
            f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            results = tracker.track_frame(frame, persist=True)
            frame_dict = {"frame_idx": f_idx, "tracks": {}}

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)

                # Assign stable slots via PlayerMerger
                slots = merger.update(boxes, track_ids)
                valid_detections += np.sum(slots != -1)

                for box, slot in zip(boxes, slots, strict=False):
                    if slot == -1:
                        continue
                    try:
                        court_pt = projector.project_bounding_box_bottom(box)
                        frame_dict["tracks"][slot] = {
                            "bbox": box.tolist(),
                            "court_pt": (float(court_pt[0]), float(court_pt[1]))
                        }
                    except RuntimeError:
                        pass

            rally_data.append(frame_dict)
            pbar.update(1)

        pbar.close()

        # If the camera angle changed (replay), the players will be projected outside
        # the court boundaries by the Master Homography and rejected (slots == -1).
        # We expect at least 1-2 players visible on average in a real rally.
        if valid_detections < total_frames * 1.0:
            logger.warning("Rally %d rejected: Geometry mismatch (Replay/Zoom). Skipping...", i+1)
            continue

        # --- PASS 2: SMOOTHING ---
        logger.info("Smoothing trajectories for Rally %d...", i+1)
        smoothed_data = smoother.process_rally(rally_data)

        # --- PASS 3: RENDERING ---
        logger.info("Playing Rally %d...", i+1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        video_writer = None
        if args.output:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"rally_{i+1:03d}.mp4"

            # Compute final dimension for video writer
            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                mm_h = int(projector.COURT_LENGTH_M * scale + margin * 2)
                mm_w = int(projector.COURT_WIDTH_M * scale + margin * 2)
                target_h = frame.shape[0]
                target_mm_w = int(mm_w * (target_h / mm_h))
                final_w = frame.shape[1] + target_mm_w

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (final_w, target_h))
                logger.info("Saving output to %s", out_path)

        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_f:
            f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            # Base 2D map
            minimap = create_birds_eye_view(projector, scale, margin)

            if f_idx in smoothed_data:
                for slot, data in smoothed_data[f_idx].items():
                    x1, y1, x2, y2 = data["bbox"]
                    cx, cy = data["court_pt"]

                    slot_colors = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 255, 0)]
                    color = slot_colors[slot]

                    # Draw on main frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"P{slot}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    bc_x, bc_y = int((x1 + x2) / 2), int(y2)
                    cv2.circle(frame, (bc_x, bc_y), 4, color, -1)

                    # Draw on minimap
                    mm_x, mm_y = court_to_img(cx, cy)
                    cv2.circle(minimap, (mm_x, mm_y), 10, color, -1)
                    cv2.putText(minimap, f"P{slot}", (mm_x - 10, mm_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Merge frames horizontally
            target_h = frame.shape[0]
            mm_h, mm_w = minimap.shape[:2]
            target_mm_w = int(mm_w * (target_h / mm_h))
            minimap_resized = cv2.resize(minimap, (target_mm_w, target_h))
            final_frame = np.hstack((frame, minimap_resized))

            if video_writer is not None:
                video_writer.write(final_frame)

            # Downscale for display
            display_h = 700
            display_w = int(final_frame.shape[1] * (display_h / final_frame.shape[0]))
            display_frame = cv2.resize(final_frame, (display_w, display_h))

            cv2.imshow(window_main, display_frame)

            # 30 ms delay, q to quit
            if cv2.waitKey(30) & 0xFF == ord("q"):
                user_quit = True
                break

        if video_writer is not None:
            video_writer.release()

        if user_quit:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
