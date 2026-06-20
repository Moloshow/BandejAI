"""Action Tube Processor for Phase 4 (Action Recognition).

Combines the ActionTubeExtractor and PoseExtractor to produce the full
skeletal sequence for each detected stroke in a rally.

Output format is compatible with PySKL's `.pkl` annotation schema.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np

from vision.action_recognition.tube_extractor import ActionTubeExtractor
from vision.action_recognition.pose_extractor import build_pose_extractor, SkeletonFrame

logger = logging.getLogger(__name__)


class ActionTubeProcessor:
    """Full pipeline: trajectories + video -> skeletal Action Tubes.

    Args:
        video_path: Path to the original rally video.
        trajectories_path: Path to the rally_XXX_trajectories.json file.
        fps: Video frame rate.
        device: Compute device.
        pose_backend: Pose extractor backend ("yolo" or "vitpose").
        tube_duration_sec: Duration of each Action Tube in seconds.
    """

    def __init__(
        self,
        video_path: Path | str,
        trajectories_path: Path | str,
        fps: float = 30.0,
        device: str = "cuda",
        pose_backend: str = "yolo",
        tube_duration_sec: float = 1.5,
    ):
        self.video_path = Path(video_path)
        self.trajectories_path = Path(trajectories_path)
        self.fps = fps
        self.device = device
        self.tube_duration_sec = tube_duration_sec

        self._extractor = ActionTubeExtractor(fps=fps, tube_duration_sec=tube_duration_sec)
        self._pose = build_pose_extractor(backend=pose_backend, device=device)
        logger.info("ActionTubeProcessor initialized with %s backend.", self._pose.backend_name)

    def process(self) -> list[dict]:
        """Detect all hits and extract a skeletal sequence for each.

        Returns:
            List of Action Tube dicts, each containing:
            - hit_frame, player_slot, tube_start, tube_end
            - frames: list of SkeletonFrame data
        """
        logger.info("Detecting hits from trajectories: %s", self.trajectories_path)
        hits = self._extractor.detect_hits(self.trajectories_path)
        logger.info("Found %d candidate hit events.", len(hits))

        if not hits:
            return []

        # Load trajectory data to get player bboxes
        with open(self.trajectories_path) as f:
            traj = json.load(f)
        players_data = traj.get("players", {})

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        action_tubes = []

        for i, hit in enumerate(hits):
            logger.info(
                "Processing stroke %d/%d: Frame %d, Player P%s",
                i + 1, len(hits), hit["hit_frame"], hit["player_slot"]
            )
            tube_frames = []
            player_slot = str(hit["player_slot"])

            for f_idx in range(hit["tube_start"], hit["tube_end"]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                # Look up player bounding box at this frame
                frame_players = players_data.get(str(f_idx), {})
                p_data = frame_players.get(player_slot)
                if p_data is None:
                    # Player not detected at this frame - add a null skeleton
                    tube_frames.append(SkeletonFrame(
                        frame_idx=f_idx,
                        keypoints=np.zeros((17, 2), dtype=np.float32),
                        confidence=np.zeros(17, dtype=np.float32),
                        bbox=[0, 0, 0, 0]
                    ))
                    continue

                bbox = p_data["bbox"]
                result = self._pose.extract(frame, bbox)

                if result is None:
                    tube_frames.append(SkeletonFrame(
                        frame_idx=f_idx,
                        keypoints=np.zeros((17, 2), dtype=np.float32),
                        confidence=np.zeros(17, dtype=np.float32),
                        bbox=bbox
                    ))
                else:
                    kp_xy, kp_conf = result
                    tube_frames.append(SkeletonFrame(
                        frame_idx=f_idx,
                        keypoints=kp_xy.astype(np.float32),
                        confidence=kp_conf.astype(np.float32),
                        bbox=bbox
                    ))

            if tube_frames:
                action_tubes.append({
                    "hit_frame": hit["hit_frame"],
                    "player_slot": hit["player_slot"],
                    "tube_start": hit["tube_start"],
                    "tube_end": hit["tube_end"],
                    "acceleration": hit["acceleration"],
                    "pose_backend": self._pose.backend_name,
                    "frames": tube_frames,
                })

        cap.release()
        logger.info("Extracted %d action tubes.", len(action_tubes))
        return action_tubes

    def save_pkl(self, action_tubes: list[dict], output_path: Path | str) -> None:
        """Save action tubes in PySKL-compatible .pkl format.

        The format follows the standard PySKL annotation schema:
        Each sample is a dict with keypoint [M, T, V, 2], keypoint_score [M, T, V],
        and label (int, -1 means unlabeled).

        Args:
            action_tubes: Output of self.process().
            output_path: Path to save the .pkl file.
        """
        samples = []
        for tube in action_tubes:
            frames = tube["frames"]
            T = len(frames)
            V = 17  # COCO keypoints

            # Shape: [M=1, T, V, 2] - one person per tube
            keypoint = np.zeros((1, T, V, 2), dtype=np.float32)
            keypoint_score = np.zeros((1, T, V), dtype=np.float32)

            for t_idx, sf in enumerate(frames):
                keypoint[0, t_idx] = sf.keypoints
                keypoint_score[0, t_idx] = sf.confidence

            samples.append({
                "hit_frame": tube["hit_frame"],
                "player_slot": tube["player_slot"],
                "tube_start": tube["tube_start"],
                "tube_end": tube["tube_end"],
                "acceleration": tube["acceleration"],
                "pose_backend": tube["pose_backend"],
                "total_frames": T,
                "img_shape": (720, 1280),  # Will be updated per video
                "keypoint": keypoint,
                "keypoint_score": keypoint_score,
                "label": -1,  # -1 = unlabeled; will be assigned during annotation
            })

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(samples, f)
        logger.info("Saved %d PySKL-format samples to %s", len(samples), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = ActionTubeProcessor(
        video_path="outputs/v17/rally_001.mp4",
        trajectories_path="outputs/v17/rally_001_trajectories.json",
        fps=25.0,
        device="cuda",
        pose_backend="yolo",
    )

    tubes = processor.process()
    processor.save_pkl(tubes, "outputs/v17/rally_001_action_tubes.pkl")

    # Quick sanity check
    import pickle
    with open("outputs/v17/rally_001_action_tubes.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"\n--- Sanity Check ---")
    print(f"Tubes extracted: {len(data)}")
    for s in data:
        kp = s["keypoint"]
        conf_mean = s["keypoint_score"].mean()
        print(f"  Frame {s['hit_frame']}: P{s['player_slot']} | Shape={kp.shape} | AvgConf={conf_mean:.2f}")
