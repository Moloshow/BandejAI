"""Visualization utilities for the BandejAI vision pipeline."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from core_math.homography.projector import CourtProjector


def court_to_img(
    x_m: float, y_m: float, projector: CourtProjector, scale: int, margin: int
) -> tuple[int, int]:
    """Convert physical court coordinates to minimap image coordinates."""
    # Invert Y so near baseline (y=0) is at the bottom of the minimap
    return (
        int(margin + x_m * scale),
        int(margin + (projector.COURT_LENGTH_M - y_m) * scale),
    )


def create_birds_eye_view(
    projector: CourtProjector, scale: int = 30, margin: int = 40
) -> NDArray[np.uint8]:
    """Create a blank 2D bird's-eye view of the court.

    Args:
        projector: Calibrated CourtProjector.
        scale: Pixels per meter.
        margin: Margin around the court in pixels.

    Returns:
        A BGR image (numpy array) of the blank court minimap.
    """
    w_px = int(projector.COURT_WIDTH_M * scale)
    l_px = int(projector.COURT_LENGTH_M * scale)

    court_img = np.ones((l_px + 2 * margin, w_px + 2 * margin, 3), dtype=np.uint8) * 30

    # Draw court background (green)
    p1 = court_to_img(0, 0, projector, scale, margin)
    p2 = court_to_img(projector.COURT_WIDTH_M, projector.COURT_LENGTH_M, projector, scale, margin)
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
        cv2.line(
            court_img,
            court_to_img(p1_c[0], p1_c[1], projector, scale, margin),
            court_to_img(p2_c[0], p2_c[1], projector, scale, margin),
            color,
            1,
        )

    # Net
    cv2.line(
        court_img,
        court_to_img(0, length / 2, projector, scale, margin),
        court_to_img(w, length / 2, projector, scale, margin),
        (0, 255, 255),
        2,
    )

    return court_img
