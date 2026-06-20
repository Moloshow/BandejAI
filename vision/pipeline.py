"""Player tracking pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from core_math.homography.projector import CourtProjector
from vision.ball_tracking.smoother import BallSmoother
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
    # PCHIP is safe enough to interpolate over 1.5 seconds (45 frames) of occlusion (e.g., passing behind the net)
    ball_smoother = BallSmoother(max_gap_frames=45)

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

        # Save raw data to identify interpolated frames later
        raw_rally_data = {d["frame_idx"]: d["tracks"] for d in rally_data}
        raw_ball_data = ball_data.copy()

        # --- PASS 2: SMOOTHING ---
        logger.info("Smoothing trajectories for Rally %d...", i + 1)
        smoothed_data = smoother.process_rally(rally_data)
        ball_data = ball_smoother.process_rally(ball_data)

        # --- PASS 3: RENDERING ---
        logger.info("Playing/Rendering Rally %d...", i + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        video_writer = None
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"rally_{i+1:03d}.mp4"
            metrics_path = out_dir / f"rally_{i+1:03d}_metrics.json"

            # --- METRICS COMPUTATION ---
            raw_ball_set = set(raw_ball_data.keys())
            smoothed_ball_set = set(ball_data.keys())
            interp_ball_count = len(smoothed_ball_set - raw_ball_set)

            # Analyze ball gaps for detailed telemetry
            ball_gaps = []
            if raw_ball_set:
                sorted_ball_frames = sorted(raw_ball_set)
                for j in range(len(sorted_ball_frames) - 1):
                    f1 = sorted_ball_frames[j]
                    f2 = sorted_ball_frames[j + 1]
                    gap_len = f2 - f1 - 1
                    if gap_len > 0:
                        # Convert pixel positions to court coordinates if possible
                        try:
                            pt1 = projector.project_point(np.array(raw_ball_data[f1]))
                            pt2 = projector.project_point(np.array(raw_ball_data[f2]))
                            c_pos1 = {"x": round(float(pt1[0]), 2), "y": round(float(pt1[1]), 2)}
                            c_pos2 = {"x": round(float(pt2[0]), 2), "y": round(float(pt2[1]), 2)}
                        except RuntimeError:
                            c_pos1 = {"x": None, "y": None}
                            c_pos2 = {"x": None, "y": None}

                        ball_gaps.append({
                            "lost_at_frame": f1,
                            "found_at_frame": f2,
                            "duration_frames": gap_len,
                            "last_seen_pixel": raw_ball_data[f1],
                            "found_pixel": raw_ball_data[f2],
                            "last_seen_court_2d": c_pos1,
                            "found_court_2d": c_pos2
                        })

            rally_metrics = {
                "rally_id": i + 1,
                "total_frames": total_frames,
                "ball": {
                    "detected_frames": len(raw_ball_set),
                    "interpolated_frames": interp_ball_count,
                    "lost_frames": total_frames - len(smoothed_ball_set),
                    "tracking_uptime_pct": round((len(raw_ball_set) / total_frames) * 100, 2) if total_frames else 0,
                    "detailed_gaps": ball_gaps
                },
                "players": {}
            }

            for slot in range(4):
                raw_count = sum(1 for f in raw_rally_data if slot in raw_rally_data[f])
                smoothed_count = sum(1 for f in smoothed_data if slot in smoothed_data[f])
                rally_metrics["players"][f"P{slot}"] = {
                    "detected_frames": raw_count,
                    "interpolated_frames": smoothed_count - raw_count,
                    "lost_frames": total_frames - smoothed_count,
                    "tracking_uptime_pct": round((raw_count / total_frames) * 100, 2) if total_frames else 0
                }

            with open(metrics_path, "w") as f:
                json.dump(rally_metrics, f, indent=4)
            logger.info("Saved metrics to %s", metrics_path)

            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                mm_h = int(projector.COURT_LENGTH_M * scale + margin * 2)
                mm_w = int(projector.COURT_WIDTH_M * scale + margin * 2)
                target_h = frame.shape[0]
                panel_h = 160
                new_mm_h = target_h - panel_h
                target_mm_w = int(mm_w * (new_mm_h / mm_h))
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
                # 1. Draw the comet trail (connect past N frames)
                trail_length = 12
                for i in range(1, trail_length):
                    f_curr = f_idx - i + 1
                    f_prev = f_idx - i
                    if f_curr in ball_data and f_prev in ball_data:
                        pt1 = ball_data[f_curr]
                        pt2 = ball_data[f_prev]
                        # The trail gets thinner as it gets older
                        thickness = max(1, int(5 * (1 - (i / trail_length))))
                        cv2.line(frame, pt2, pt1, (0, 255, 255), thickness)

                # 2. Draw the current ball position
                bx, by = ball_data[f_idx]
                cv2.circle(frame, (bx, by), 5, (0, 255, 255), -1)  # Yellow circle
                cv2.circle(frame, (bx, by), 7, (0, 0, 0), 2)  # Black border

                try:
                    # Project ball to minimap (assuming it is near the ground for now)
                    court_pt = projector.project_point(np.array([bx, by]))
                    
                    # In Padel, the court is enclosed by glass at 10x20m.
                    # A projection outside usually means Z > 0 (e.g., an airborne Lob).
                    is_out = court_pt[0] < -0.5 or court_pt[0] > 10.5 or court_pt[1] < -0.5 or court_pt[1] > 20.5
                    
                    if is_out:
                        # Clamp to the physical court boundary (the glass)
                        clamped_x = np.clip(court_pt[0], 0.0, 10.0)
                        clamped_y = np.clip(court_pt[1], 0.0, 20.0)
                        mm_bx, mm_by = court_to_img(float(clamped_x), float(clamped_y), projector, scale, margin)
                        
                        # Draw as a hollow orange circle to signify "Airborne / Z-axis uncertainty"
                        cv2.circle(minimap, (mm_bx, mm_by), 6, (0, 165, 255), 2)
                    else:
                        mm_bx, mm_by = court_to_img(float(court_pt[0]), float(court_pt[1]), projector, scale, margin)
                        cv2.circle(minimap, (mm_bx, mm_by), 6, (0, 255, 255), -1)
                        cv2.circle(minimap, (mm_bx, mm_by), 8, (0, 0, 0), 2)
                except RuntimeError:
                    pass

            target_h = frame.shape[0]
            mm_h, mm_w = minimap.shape[:2]
            
            # Create a status panel at the bottom of the minimap
            panel_h = 160
            new_mm_h = target_h - panel_h
            target_mm_w = int(mm_w * (new_mm_h / mm_h))  # Preserve aspect ratio with new height
            
            minimap_resized = cv2.resize(minimap, (target_mm_w, new_mm_h))
            panel = np.zeros((panel_h, target_mm_w, 3), dtype=np.uint8)

            # Draw Status Header
            cv2.putText(panel, "LIVE TRACKING STATUS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.line(panel, (10, 40), (target_mm_w - 10, 40), (100, 100, 100), 1)

            # Ball Status
            ball_is_interp = f_idx in ball_data and f_idx not in raw_ball_data
            if f_idx not in ball_data:
                b_text, b_color = "BALL: Lost", (50, 50, 50)
            elif ball_is_interp:
                b_text, b_color = "BALL: Interpolated", (0, 165, 255) # Orange
            else:
                b_text, b_color = "BALL: Tracked", (0, 255, 0) # Green

            cv2.putText(panel, b_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_color, 2)

            # Players Status
            if f_idx in smoothed_data:
                for slot in smoothed_data[f_idx]:
                    p_is_interp = slot not in raw_rally_data.get(f_idx, {})
                    p_text = f"P{slot}: Interp" if p_is_interp else f"P{slot}: Tracked"
                    p_color = (0, 165, 255) if p_is_interp else (0, 255, 0)

                    row = slot // 2
                    col = slot % 2
                    px = 10 + col * int(target_mm_w / 2)
                    py = 110 + row * 30
                    cv2.putText(panel, p_text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 1 if p_is_interp else 2)

            # Stack minimap and panel
            minimap_final = np.vstack((minimap_resized, panel))
            final_frame = np.hstack((frame, minimap_final))

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
