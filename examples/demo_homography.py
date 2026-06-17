"""Interactive homography calibration demo.

Click on the 15 court keypoints in the image to compute the homography matrix
and project points from image-space to court-space (bird's-eye view).

Usage:
    python examples/demo_homography.py --image data/frame_001.jpg
    python examples/demo_homography.py --video data/sample_match.mp4.webm --frame 30

Controls (when the image window is focused):
    - Left click:  Select the next keypoint
    - Right click: Undo last keypoint
    - r:           Reset all keypoints
    - q / ESC:     Quit

After all 15 keypoints are selected, two windows appear:
    1. The original image with the court overlay (lines drawn via homography)
    2. A 2D bird's-eye view of the court with projected points

Requirements:
    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core_math.homography.projector import CourtProjector  # noqa: E402

logger = logging.getLogger("bandejai.demo")

# Colors in BGR format
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)


class HomographyDemo:
    """Interactive tool for calibrating and testing court homography."""

    WINDOW_IMAGE: ClassVar[str] = "Click keypoints (q=quit, r=reset)"
    WINDOW_COURT: ClassVar[str] = "Bird's-eye view (q=quit)"

    # Keypoint names for user guidance (15 = 5 rows x 3 columns)
    KEYPOINT_NAMES: ClassVar[list[str]] = [
        "Near Baseline Left",
        "Near Baseline Center",
        "Near Baseline Right",
        "Near Service Left",
        "Near Service Center",
        "Near Service Right",
        "Net Left",
        "Net Center",
        "Net Right",
        "Far Service Left",
        "Far Service Center",
        "Far Service Right",
        "Far Baseline Left",
        "Far Baseline Center",
        "Far Baseline Right",
    ]

    def __init__(self, image: NDArray[np.uint8], points_mode: int = 15) -> None:
        self.projector = CourtProjector()
        self.image = image.copy()
        self.image_display = image.copy()
        self.clicked_points: list[tuple[int, int]] = []

        # 0-based indices of the full 15-point layout
        # 0: Near Base L,  1: Near Base C,  2: Near Base R
        # 3: Near Serv L,  4: Near Serv C,  5: Near Serv R
        # 6: Net L,        7: Net C,        8: Net R
        # 9: Far Serv L,   10: Far Serv C,  11: Far Serv R
        # 12: Far Base L,  13: Far Base C,  14: Far Base R
        if points_mode == 15:
            self.active_indices = list(range(15))
        elif points_mode == 12:
            # Without base centers and net center (1, 7, 13)
            # Only points with physical intersections are kept!
            self.active_indices = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]
        elif points_mode == 10:
            # Without any centers (left and right columns only)
            self.active_indices = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14]
        elif points_mode == 6:
            # Without any centers and without service lines (corners + net)
            self.active_indices = [0, 2, 6, 8, 12, 14]
        elif points_mode == 4:
            # 4 outer corners only (minimum for homography)
            self.active_indices = [0, 2, 12, 14]
        else:
            raise ValueError(f"Unsupported points mode: {points_mode}")

        self.active_names = [self.KEYPOINT_NAMES[i] for i in self.active_indices]
        self.court_template = self.projector.court_keypoints_template[self.active_indices]
        self.num_points = len(self.active_indices)

    def run(self) -> None:
        """Run the interactive calibration loop."""
        cv2.namedWindow(self.WINDOW_IMAGE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_IMAGE, self._mouse_callback)

        logger.info(f"Starting interactive calibration. Click {self.num_points} keypoints.")

        while True:
            self._draw_instructions()
            cv2.imshow(self.WINDOW_IMAGE, self.image_display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("r"):
                self._reset()

        cv2.destroyAllWindows()

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """Handle mouse clicks for keypoint selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicked_points) < self.num_points:
                self.clicked_points.append((x, y))
                logger.info(
                    "Keypoint %d/%d: %s at (%d, %d)",
                    len(self.clicked_points),
                    self.num_points,
                    self.active_names[len(self.clicked_points) - 1],
                    x,
                    y,
                )

                if len(self.clicked_points) == self.num_points:
                    self._finalize_calibration()

        elif event == cv2.EVENT_RBUTTONDOWN and self.clicked_points:
            removed = self.clicked_points.pop()
            logger.info("Removed last keypoint: %s", removed)
            self.image_display = self.image.copy()

    def _draw_instructions(self) -> None:
        """Draw keypoint indices and instructions on the image."""
        # Draw clicked points
        for i, (x, y) in enumerate(self.clicked_points):
            cv2.circle(self.image_display, (x, y), 5, COLOR_GREEN, -1)
            cv2.putText(
                self.image_display,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_YELLOW,
                2,
            )

        # Draw instruction text at top
        idx = len(self.clicked_points)
        if idx < self.num_points:
            name = self.active_names[idx] if idx < len(self.active_names) else "?"
            text = f"Click #{idx + 1}/{self.num_points}: {name}"
        else:
            text = "All keypoints selected! Check bird's-eye view."

        cv2.rectangle(self.image_display, (0, 0), (self.image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            self.image_display,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_WHITE,
            2,
        )

    def _finalize_calibration(self) -> None:
        """Compute homography and show the bird's-eye view."""
        image_pts = np.array(self.clicked_points, dtype=np.float64)
        court_pts = self.court_template

        try:
            self.projector.compute_homography(image_pts, court_pts)
        except (ValueError, RuntimeError) as e:
            logger.error("Calibration failed: %s", e)
            return

        logger.info("Homography computed successfully!")

        # Draw court overlay on the image
        self._draw_court_overlay()

        # Show bird's-eye view
        self._show_birds_eye_view()

    def _draw_court_overlay(self) -> None:
        """Draw court lines on the image by inverse-projecting court template."""
        if not self.projector.is_calibrated or self.projector.homography is None:
            return

        H_inv = np.linalg.inv(self.projector.homography)

        # Define court line segments in court-space (x, y) in meters
        w = self.projector.COURT_WIDTH_M
        length = self.projector.COURT_LENGTH_M
        s = self.projector.SERVICE_LINE_DIST_M

        lines = [
            # Baseline (near)
            [[0, 0], [w, 0]],
            # Baseline (far)
            [[0, length], [w, length]],
            # Sidelines (left and right)
            [[0, 0], [0, length]],
            [[w, 0], [w, length]],
            # Service lines (near half)
            [[0, s], [w, s]],
            # Service lines (far half)
            [[0, length - s], [w, length - s]],
            # Center service line (near half)
            [[w / 2, s], [w / 2, length / 2]],
            # Center service line (far half)
            [[w / 2, length / 2], [w / 2, length - s]],
            # Net
            [[0, length / 2], [w, length / 2]],
        ]

        for p1_court, p2_court in lines:
            pts = np.array([p1_court, p2_court], dtype=np.float32).reshape(-1, 1, 2)
            pts_img = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
            p1 = tuple(pts_img[0].astype(int))
            p2 = tuple(pts_img[1].astype(int))
            cv2.line(self.image_display, p1, p2, COLOR_BLUE, 2)

    def _show_birds_eye_view(self) -> None:
        """Create and display a 2D bird's-eye view of the court."""
        scale = 30  # pixels per meter
        w_px = int(self.projector.COURT_WIDTH_M * scale)
        l_px = int(self.projector.COURT_LENGTH_M * scale)
        margin = 40

        court_img = np.ones((l_px + 2 * margin, w_px + 2 * margin, 3), dtype=np.uint8) * 30

        def court_to_img(x_m: float, y_m: float) -> tuple[int, int]:
            """Convert court-space (meters) to bird's-eye image pixels."""
            return int(margin + x_m * scale), int(margin + y_m * scale)

        # Draw court background (green)
        p1 = court_to_img(0, 0)
        p2 = court_to_img(self.projector.COURT_WIDTH_M, self.projector.COURT_LENGTH_M)
        cv2.rectangle(court_img, p1, p2, (34, 89, 56), -1)

        # Draw clicked points projected to court-space
        image_pts = np.array(self.clicked_points, dtype=np.float64)
        court_pts = self.projector.project_points(image_pts)

        for i, (cx, cy) in enumerate(court_pts):
            px, py = court_to_img(cx, cy)
            cv2.circle(court_img, (px, py), 8, COLOR_RED, -1)
            cv2.putText(
                court_img,
                str(i + 1),
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_YELLOW,
                2,
            )

        # Draw court lines
        w = self.projector.COURT_WIDTH_M
        length = self.projector.COURT_LENGTH_M
        s = self.projector.SERVICE_LINE_DIST_M

        court_lines = [
            ([0, 0], [w, 0], COLOR_WHITE),
            ([0, length], [w, length], COLOR_WHITE),
            ([0, 0], [0, length], COLOR_WHITE),
            ([w, 0], [w, length], COLOR_WHITE),
            ([0, s], [w, s], COLOR_WHITE),
            ([0, length - s], [w, length - s], COLOR_WHITE),
            ([w / 2, s], [w / 2, length / 2], COLOR_WHITE),
            ([w / 2, length / 2], [w / 2, length - s], COLOR_WHITE),
        ]
        for p1_c, p2_c, color in court_lines:
            cv2.line(court_img, court_to_img(*p1_c), court_to_img(*p2_c), color, 1)

        # Net (dashed-ish, thicker)
        cv2.line(
            court_img,
            court_to_img(0, length / 2),
            court_to_img(w, length / 2),
            COLOR_YELLOW,
            2,
        )

        cv2.imshow(self.WINDOW_COURT, court_img)
        logger.info("Bird's-eye view displayed. Press 'q' to quit.")

    def _reset(self) -> None:
        """Reset all clicked points and the display."""
        self.clicked_points.clear()
        self.image_display = self.image.copy()
        self.projector = CourtProjector()
        cv2.destroyWindow(self.WINDOW_COURT) if cv2.getWindowProperty(
            self.WINDOW_COURT, cv2.WND_PROP_VISIBLE
        ) >= 1 else None
        logger.info("Reset. Click keypoints again.")


def extract_frame_from_video(video_path: str, frame_num: int = 0) -> NDArray[np.uint8]:
    """Extract a specific frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Cannot read frame {frame_num} from {video_path}")

    return np.asarray(frame, dtype=np.uint8)


def main() -> None:
    """Entry point for the homography demo."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Interactive homography calibration demo")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to image file (JPG/PNG)")
    group.add_argument("--video", type=str, help="Path to video file (MP4/WebM)")
    parser.add_argument("--frame", type=int, default=30, help="Frame number (for --video)")
    parser.add_argument(
        "--points",
        type=int,
        choices=[15, 12, 10, 6, 4],
        default=15,
        help="Number of keypoints to click (15=full, 12=no pure centers, 10=edges only, 6=corners+net, 4=outer corners)",
    )
    args = parser.parse_args()

    if args.image:
        raw = cv2.imread(args.image)
        if raw is None:
            logger.error("Cannot read image: %s", args.image)
            sys.exit(1)
        image = np.asarray(raw, dtype=np.uint8)
    else:
        image = extract_frame_from_video(args.video, args.frame)

    logger.info("Image loaded: %dx%d", image.shape[1], image.shape[0])
    demo = HomographyDemo(image, points_mode=args.points)
    demo.run()


if __name__ == "__main__":
    main()


