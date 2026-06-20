"""Player tracking pipeline orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core_math.homography.projector import CourtProjector
from vision.ball_tracking.smoother import BallSmoother
from vision.ball_tracking.tracker import BallTracker
from vision.player_tracking.merger import PlayerMerger
from vision.player_tracking.smoother import TrajectorySmoother
from vision.player_tracking.tracker import PlayerTracker
from vision.visualization import court_to_img, create_birds_eye_view
from vision.action_recognition.tube_extractor import ActionTubeExtractor
from vision.action_recognition.pose_extractor import build_pose_extractor

logger = logging.getLogger(__name__)

# COCO skeleton connections for skeleton overlay drawing
_SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
    (0, 5), (0, 6),                              # head-shoulders
]


def _draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, confidence: np.ndarray,
                   color: tuple = (0, 255, 128), threshold: float = 0.3) -> None:
    """Draw skeleton keypoints and edges on a frame in-place."""
    for j1, j2 in _SKELETON_EDGES:
        if confidence[j1] > threshold and confidence[j2] > threshold:
            p1 = (int(keypoints[j1, 0]), int(keypoints[j1, 1]))
            p2 = (int(keypoints[j2, 0]), int(keypoints[j2, 1]))
            cv2.line(frame, p1, p2, color, 2)
    for j in range(len(keypoints)):
        if confidence[j] > threshold:
            cx, cy = int(keypoints[j, 0]), int(keypoints[j, 1])
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 4, color, 1)


def _render_loading_screen(
    window: str,
    phase_num: int,
    phase_name: str,
    rally_num: int,
    total_rallies: int,
    progress: float = 0.0,
    detail: str = "",
    device: str | None = None,
) -> None:
    """Draw a unified loading screen for all pipeline phases.

    NOTE: cv2.putText does not support Unicode/accented characters on Windows.
    All strings passed here must be plain ASCII.
    """
    canvas = np.zeros((400, 800, 3), dtype=np.uint8)
    W = 800  # canvas width

    # --- Phase pill ---
    phase_colors = {1: (0, 220, 80), 2: (0, 165, 255), 3: (0, 220, 220)}
    p_color = phase_colors.get(phase_num, (200, 200, 200))
    cv2.rectangle(canvas, (30, 28), (150, 58), p_color, -1)
    cv2.putText(canvas, f"PASS {phase_num}", (38, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

    # --- Phase name ---
    cv2.putText(canvas, phase_name, (165, 51),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # --- Rally indicator ---
    cv2.putText(canvas, f"Rally {rally_num} / {total_rallies}", (30, 93),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

    # --- Device badge ---
    if device:
        d_color = (0, 220, 80) if device.lower() == "cuda" else (0, 165, 255)
        cv2.putText(canvas, f"Device: {device.upper()}", (540, 93),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, d_color, 1)

    # --- Separator ---
    cv2.line(canvas, (30, 108), (W - 30, 108), (60, 60, 60), 1)

    # --- Progress bar (leave 70px on right for % label) ---
    bar_x, bar_y, bar_h = 30, 148, 26
    pct_label_w = 65  # reserved width for "100%" text
    bar_w = W - 30 - bar_x - pct_label_w  # = 675
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    filled = int(bar_w * max(0.0, min(1.0, progress)))
    if filled > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), p_color, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    # --- Percentage label (right of bar, always visible) ---
    pct_text = f"{int(progress * 100)}%"
    cv2.putText(canvas, pct_text, (bar_x + bar_w + 10, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    # --- Detail text ---
    if detail:
        cv2.putText(canvas, detail, (30, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow(window, canvas)
    cv2.waitKey(1)


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

    # Detect devices once for all passes
    _inference_device = "cuda" if torch.cuda.is_available() else "cpu"  # YOLO, TrackNet, Pose
    _cpu_device = "cpu"  # Smoothing is always numpy/pandas (CPU)
    logger.info("Inference device: %s", _inference_device.upper())

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
        pbar = tqdm(total=total_frames, desc=f"Pass 1 - Tracking Rally {i+1}")

        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_f:
            f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if not ret:
                break

            results = tracker.track_frame(frame, persist=True)
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

            if show_window and pbar.n % 5 == 0:
                _render_loading_screen(
                    window_main, 1, "Tracking & Detection",
                    i + 1, len(rallies),
                    progress=pbar.n / max(1, total_frames),
                    detail=f"Frame {pbar.n} / {total_frames}",
                    device=_inference_device,
                )

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
        if show_window:
            _render_loading_screen(
                window_main, 2, "Smoothing & Filtering",
                i + 1, len(rallies),
                progress=0.5,
                detail="PCHIP interpolation + Median filter...",
                device=_cpu_device,
            )

        smoothed_data = smoother.process_rally(rally_data)
        ball_data, ball_metadata = ball_smoother.process_rally(ball_data, return_metadata=True)

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
                    "hotspots_masked": ball_metadata.get("hotspots", []),
                    "static_frames_dropped": ball_metadata.get("static_frames_dropped", 0),
                    "outliers_dropped": ball_metadata.get("outliers_dropped", 0),
                    "filter_actions": ball_metadata.get("filter_actions", []),
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

            # Export full trajectories for Phase 4 (Action Recognition)
            trajectories_path = out_dir / f"rally_{i+1:03d}_trajectories.json"
            trajectories_data = {
                "ball": {int(k): [float(v[0]), float(v[1])] for k, v in ball_data.items()},
                "players": {
                    int(f_idx): {
                        slot: {"bbox": [float(x) for x in data["bbox"]], "court_pt": [float(x) for x in data["court_pt"]]}
                        for slot, data in frame_data.items()
                    }
                    for f_idx, frame_data in smoothed_data.items()
                }
            }
            with open(trajectories_path, "w") as f:
                json.dump(trajectories_data, f, indent=2)
            logger.info("Saved full trajectories to %s", trajectories_path)

            # --- Phase 4: Detect Stroke Events for Visualization ---
            _tube_extractor = ActionTubeExtractor(fps=cap.get(cv2.CAP_PROP_FPS) or 25.0)
            detected_hits = _tube_extractor.detect_hits(trajectories_path)
            logger.info("Detected %d stroke events for overlay rendering.", len(detected_hits))
            for h in detected_hits:
                logger.info(
                    "  Stroke @ Frame %d - Player P%s (Acc=%.1f px/f²)",
                    h["hit_frame"], h["player_slot"], h["acceleration"]
                )
            # Build fast lookup: frame_idx -> hit metadata (for O(1) render-loop check)
            hit_by_frame = {h["hit_frame"]: h for h in detected_hits}
            # Build tube membership set: all frames within any action tube
            tube_frames_map: dict[int, dict] = {}  # frame_idx -> hit metadata
            for h in detected_hits:
                for tf in range(h["tube_start"], h["tube_end"]):
                    tube_frames_map[tf] = h

            # --- PASS 3: POSE EXTRACTION ---
            # Pre-compute ALL skeletons before the render loop so there is zero
            # inference overhead during display. Render loop (Pass 4) becomes pure drawing.
            _pose_device = _inference_device  # Same GPU as YOLO/TrackNet
            logger.info("Extracting pose keypoints for Rally %d (Pass 3) on %s...", i + 1, _pose_device.upper())
            _pose_extractor = build_pose_extractor(backend="yolo", device=_pose_device)

            # skeleton_cache[frame_idx][slot] = (keypoints [17,2], confidence [17])
            skeleton_cache: dict[int, dict] = {}
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            pbar_pose = tqdm(total=total_frames, desc=f"Pass 3 - Pose Rally {i+1} [{_pose_device.upper()}]")

            if show_window:
                _render_loading_screen(
                    window_main, 3, "Pose Extraction",
                    i + 1, len(rallies),
                    progress=0.0,
                    detail="Initializing...",
                    device=_pose_device,
                )

            while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_f:
                pf_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, pose_frame = cap.read()
                if not ret:
                    break
                if pf_idx in smoothed_data:
                    skeleton_cache[pf_idx] = {}
                    for slot, data in smoothed_data[pf_idx].items():
                        pose_result = _pose_extractor.extract(pose_frame, data["bbox"])
                        if pose_result is not None:
                            skeleton_cache[pf_idx][slot] = pose_result
                pbar_pose.update(1)

                if show_window and pbar_pose.n % 10 == 0:
                    _render_loading_screen(
                        window_main, 3, "Pose Extraction",
                        i + 1, len(rallies),
                        progress=pbar_pose.n / max(1, total_frames),
                        detail=f"Frame {pbar_pose.n} / {total_frames}",
                        device=_pose_device,
                    )
            pbar_pose.close()
            logger.info("Pass 3 complete: %d frames cached (%s).", len(skeleton_cache), _pose_device.upper())

            # --- PASS 4: RENDERING ---
            logger.info("Rendering Rally %d (Pass 4)...", i + 1)

            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                mm_h = int(projector.COURT_LENGTH_M * scale + margin * 2)
                mm_w = int(projector.COURT_WIDTH_M * scale + margin * 2)
                target_h = frame.shape[0]
                panel_h = 140
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

            # Phase 4: Check if this frame is an impact frame or inside an action tube
            is_hit_frame = f_idx in hit_by_frame
            active_tube = tube_frames_map.get(f_idx)
            active_hitter_slot = int(active_tube["player_slot"]) if active_tube else None

            if f_idx in smoothed_data:
                for slot, data in smoothed_data[f_idx].items():
                    x1, y1, x2, y2 = data["bbox"]
                    cx, cy = data["court_pt"]

                    slot_colors = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (255, 255, 0)]
                    color = slot_colors[slot]

                    # Highlight the active hitter with a thicker box
                    thickness = 4 if slot == active_hitter_slot else 2
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    cv2.putText(frame, f"P{slot}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    bc_x, bc_y = int((x1 + x2) / 2), int(y2)
                    cv2.circle(frame, (bc_x, bc_y), 4, color, -1)

                    mm_x, mm_y = court_to_img(cx, cy, projector, scale, margin)
                    cv2.circle(minimap, (mm_x, mm_y), 10, color, -1)
                    cv2.putText(minimap, f"P{slot}", (mm_x - 10, mm_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Draw skeleton from pre-computed cache (zero inference overhead)
                    frame_skeletons = skeleton_cache.get(f_idx, {})
                    if slot in frame_skeletons:
                        kp_xy, kp_conf = frame_skeletons[slot]
                        if is_hit_frame and slot == active_hitter_slot:
                            skel_color = (0, 0, 255)  # Red flash on hitter at impact
                        else:
                            skel_color = color  # Match player slot color
                        _draw_skeleton(frame, kp_xy, kp_conf, color=skel_color)

            # Flash overlay on exact impact frame (or up to 5 frames after)
            recent_hit = None
            for h_frame in range(max(0, f_idx - 5), f_idx + 1):
                if h_frame in hit_by_frame:
                    recent_hit = hit_by_frame[h_frame]
                    break

            if recent_hit:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 30)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.putText(frame, f"STROKE  P{recent_hit['player_slot']}",
                            (frame.shape[1] // 2 - 160, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

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
            panel_h = 140
            new_mm_h = target_h - panel_h
            target_mm_w = int(mm_w * (new_mm_h / mm_h))  # Preserve aspect ratio with new height
            
            minimap_resized = cv2.resize(minimap, (target_mm_w, new_mm_h))
            panel = np.zeros((panel_h, target_mm_w, 3), dtype=np.uint8)

            # Ball Status
            ball_is_interp = f_idx in ball_data and f_idx not in raw_ball_data
            if f_idx not in ball_data:
                b_text, b_color = "BALL: Lost", (50, 50, 50)
            elif ball_is_interp:
                b_text, b_color = "BALL: Interpolated", (0, 165, 255)
            else:
                b_text, b_color = "BALL: Tracked", (0, 255, 0)
            cv2.putText(panel, b_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, b_color, 2)

            # Stroke Event status (Phase 4)
            past_hits = [h for h in detected_hits if h["hit_frame"] <= f_idx]
            current_stroke_count = len(past_hits)
            total_strokes = len(detected_hits)
            last_hitter = f"P{past_hits[-1]['player_slot']}" if past_hits else "-"
            
            # Always show the stroke counter
            count_text = f"Strokes: {current_stroke_count}/{total_strokes}  |  Last: {last_hitter}"
            cv2.putText(panel, count_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 2)
            
            # Show active event (if any) on a separate line
            event_text = ""
            if recent_hit:
                event_text = f"STROKE  P{recent_hit['player_slot']}"
                event_color = (0, 0, 255)
            elif active_tube is not None:
                event_text = f"Tube: P{active_tube['player_slot']} (pose active)"
                event_color = (0, 165, 255)
            
            if event_text:
                cv2.putText(panel, event_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, event_color, 2)

            # Players Status
            if f_idx in smoothed_data:
                for slot in smoothed_data[f_idx]:
                    p_is_interp = slot not in raw_rally_data.get(f_idx, {})
                    p_text = f"P{slot}: Interp" if p_is_interp else f"P{slot}: Tracked"
                    p_color = (0, 165, 255) if p_is_interp else (0, 255, 0)

                    row = slot // 2
                    col = slot % 2
                    px = 10 + col * int(target_mm_w / 2)
                    py = 105 + row * 26
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

                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                wait_delay = max(1, int(1000 / fps))
                if cv2.waitKey(wait_delay) & 0xFF == ord("q"):
                    user_quit = True
                    break

        if video_writer is not None:
            video_writer.release()

        if user_quit:
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
