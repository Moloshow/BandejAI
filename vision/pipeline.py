"""Player tracking pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from core_math.homography.projector import CourtProjector
from vision.ball_tracking.tracker import BallTracker
from vision.player_tracking.merger import PlayerMerger
from vision.player_tracking.smoother import TrajectorySmoother
from vision.player_tracking.tracker import PlayerTracker
from vision.visualization import court_to_img, create_birds_eye_view

logger = logging.getLogger(__name__)


def process_player_tracking(
    video_path: Path | str,
    rallies: list[list[int]],
    projector: CourtProjector,
    output_dir: Path | str | None = None,
    show_window: bool = True,
) -> None:
    """Run the 3-pass tracking pipeline on the given rallies.

    Args:
        video_path: Path to the input video.
        rallies: List of [start_frame, end_frame] pairs.
        projector: Calibrated CourtProjector.
        output_dir: Directory to save the MP4 exports. If None, won't save.
        show_window: Whether to show the UI dashboard via cv2.imshow.
    """
    video_path_str = str(video_path)
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path_str}")

    logger.info("Initializing PlayerTracker...")
    tracker = PlayerTracker("yolov8n.pt")
    logger.info("Initializing BallTracker...")
    ball_tracker = BallTracker()
    smoother = TrajectorySmoother(max_gap_frames=60)

    window_main = "Padelytics Tracking Dashboard"
    if show_window:
        cv2.namedWindow(window_main, cv2.WINDOW_AUTOSIZE)

    scale = 30
    margin = 40

    user_quit = False

    for i, (start_f, end_f) in enumerate(rallies):
        logger.info("=== Processing Rally %d/%d (Frames %d -> %d) ===", i + 1, len(rallies), start_f, end_f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        ball_tracker.reset()

        merger = PlayerMerger(projector, max_lost_frames=60)
        rally_data = []
        ball_data = {}  # Store ball coordinates by frame index
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
            # update() returns (pos, target_idx) where target_idx is the newest
            # frame (t) using [t, t-1, t-2] ordering to eliminate prediction lag.
            ball_pos, ball_frame_idx = ball_tracker.update(frame, f_idx)
            if ball_pos is not None and ball_frame_idx is not None:
                ball_data[ball_frame_idx] = ball_pos

            frame_dict = {"frame_idx": f_idx, "tracks": {}}

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)

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

        # Replay/Zoom rejection
        if valid_detections < total_frames * 1.0:
            logger.warning("Rally %d rejected: Geometry mismatch (Replay/Zoom). Skipping...", i + 1)
            continue

        # --- PASS 2: SMOOTHING ---
        logger.info("Smoothing trajectories for Rally %d...", i + 1)
        smoothed_data = smoother.process_rally(rally_data)

        # --- PASS 3: RENDERING ---
        logger.info("Playing/Rendering Rally %d...", i + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        video_writer = None
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"rally_{i+1:03d}.mp4"

            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                mm_h = int(projector.COURT_LENGTH_M * scale + margin * 2)
                mm_w = int(projector.COURT_WIDTH_M * scale + margin * 2)
                target_h = frame.shape[0]
                target_mm_w = int(mm_w * (target_h / mm_h))
                final_w = frame.shape[1] + target_mm_w

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (final_w, target_h))
                logger.info("Saving output to %s", out_path)

        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_f:
            f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            minimap = create_birds_eye_view(projector, scale, margin)

            if f_idx in smoothed_data:
                for slot, data in smoothed_data[f_idx].items():
                    x1, y1, x2, y2 = data["bbox"]
                    cx, cy = data["court_pt"]

                    slot_colors = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 255, 0)]
                    color = slot_colors[slot]

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"P{slot}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    bc_x, bc_y = int((x1 + x2) / 2), int(y2)
                    cv2.circle(frame, (bc_x, bc_y), 4, color, -1)

                    mm_x, mm_y = court_to_img(cx, cy, projector, scale, margin)
                    cv2.circle(minimap, (mm_x, mm_y), 10, color, -1)
                    cv2.putText(minimap, f"P{slot}", (mm_x - 10, mm_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if f_idx in ball_data:
                bx, by = ball_data[f_idx]
                cv2.circle(frame, (bx, by), 6, (0, 255, 255), -1)  # Yellow circle
                cv2.circle(frame, (bx, by), 8, (0, 0, 0), 2)  # Black border

                try:
                    # Project ball to minimap (assuming it is near the ground for now)
                    court_pt = projector.project_point(np.array([bx, by]))
                    mm_bx, mm_by = court_to_img(float(court_pt[0]), float(court_pt[1]), projector, scale, margin)
                    cv2.circle(minimap, (mm_bx, mm_by), 6, (0, 255, 255), -1)
                    cv2.circle(minimap, (mm_bx, mm_by), 8, (0, 0, 0), 2)
                except RuntimeError:
                    pass

            target_h = frame.shape[0]
            mm_h, mm_w = minimap.shape[:2]
            target_mm_w = int(mm_w * (target_h / mm_h))
            minimap_resized = cv2.resize(minimap, (target_mm_w, target_h))
            final_frame = np.hstack((frame, minimap_resized))

            if video_writer is not None:
                video_writer.write(final_frame)

            if show_window:
                display_h = 700
                display_w = int(final_frame.shape[1] * (display_h / final_frame.shape[0]))
                display_frame = cv2.resize(final_frame, (display_w, display_h))
                cv2.imshow(window_main, display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    user_quit = True
                    break

        if video_writer is not None:
            video_writer.release()

        if user_quit:
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
