"""Homography-based court projection.

Transforms image coordinates (u, v) to 2D court-space (x, y) using a
perspective homography matrix computed from court keypoints.

This is a deterministic, explainable module - no neural networks involved.
The homography matrix H is computed once from manually-annotated keypoints
(see main.py: run_homography_init) and then applied to every frame.

Mathematical formulation:
    [[x], [y], [1]] = c * H * [[u], [v], [1]]

where c is a scaling factor to normalize the homogeneous coordinate back to 1.

Status: Phase 0 (skeleton). Implementation deferred to Phase 1.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("bandejai.homography")

# Type aliases for clarity (float64 arrays)
ImagePoint = NDArray[np.float64]  # shape (2,) or (N, 2)
CourtPoint = NDArray[np.float64]  # shape (2,) or (N, 2)


class CourtProjector:
    """Projects image-space pixel coordinates onto a 2D padel court model.

    The court model uses a coordinate system in meters, with the origin at
    the center of the baseline (outside the glass wall). The x-axis runs
    along the baseline, the y-axis runs toward the net.

    Attributes:
        homography: 3x3 homography matrix mapping image -> court.
                    None until :meth:`compute_homography` is called.
        court_dimensions: (width, length) of a standard padel court in meters.
                          Default: (10, 20) per FIP regulations.

    """

    # Standard padel court dimensions (FIP regulations), in meters
    COURT_WIDTH_M: ClassVar[float] = 10.0
    COURT_LENGTH_M: ClassVar[float] = 20.0
    NET_HEIGHT_M: ClassVar[float] = 0.88

    # Expected number of keypoints for manual annotation
    NUM_KEYPOINTS: ClassVar[int] = 12

    def __init__(self) -> None:
        """Initialize the projector without a homography matrix."""
        self._homography: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def homography(self) -> NDArray[np.float64] | None:
        """Return the current homography matrix, or None if not computed."""
        return self._homography

    @property
    def is_calibrated(self) -> bool:
        """Return True if the homography matrix has been computed."""
        return self._homography is not None

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def compute_homography(
        self,
        image_points: NDArray[np.float64],
        court_points: NDArray[np.float64],
    ) -> None:
        """Compute the homography matrix from paired keypoints.

        Uses OpenCV's :func:`cv2.findHomography` with RANSAC for robustness
        against slight annotation errors.

        Args:
            image_points: Array of shape (N, 2) with pixel coordinates (u, v).
            court_points: Array of shape (N, 2) with court-space coordinates
                         (x, y) in meters.

        Raises:
            ValueError: If fewer than 4 point pairs are provided or if shapes
                       mismatch.
            RuntimeError: If OpenCV fails to compute a valid homography.

        Todo:
            - Phase 1: Implement using cv2.findHomography(image, court, RANSAC).
            - Add validation that court_points lie within court bounds.

        """
        if len(image_points) < 4 or len(court_points) < 4:
            raise ValueError(
                f"At least 4 point pairs required, got "
                f"image={len(image_points)}, court={len(court_points)}"
            )
        if image_points.shape != court_points.shape:
            raise ValueError(
                f"Shape mismatch: image={image_points.shape}, court={court_points.shape}"
            )

        # TODO(Phase 1): H, mask = cv2.findHomography(image_points, court_points, cv2.RANSAC, 5.0)
        # TODO(Phase 1): Validate H is not None
        # TODO(Phase 1): self._homography = H
        logger.warning("compute_homography() not yet implemented (Phase 1 TODO)")

    # ------------------------------------------------------------------ #
    # Projection
    # ------------------------------------------------------------------ #
    def project_point(self, image_point: ImagePoint) -> CourtPoint:
        """Project a single image-space point to 2D court-space.

        Args:
            image_point: Array of shape (2,) with pixel coordinates (u, v).

        Returns:
            Array of shape (2,) with court-space coordinates (x, y) in meters.

        Raises:
            RuntimeError: If the homography has not been computed yet.

        Todo:
            - Phase 1: Implement using cv2.perspectiveTransform.

        """
        if not self.is_calibrated:
            raise RuntimeError("Homography not computed. Call compute_homography() first.")

        # TODO(Phase 1): point = np.array([[image_point]], dtype=np.float64)
        # TODO(Phase 1): court = cv2.perspectiveTransform(point, self._homography)
        # TODO(Phase 1): return court[0, 0]
        logger.warning("project_point() not yet implemented (Phase 1 TODO)")
        return np.zeros(2, dtype=np.float64)

    def project_points(self, image_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project multiple image-space points to 2D court-space.

        Args:
            image_points: Array of shape (N, 2) with pixel coordinates.

        Returns:
            Array of shape (N, 2) with court-space coordinates in meters.

        Raises:
            RuntimeError: If the homography has not been computed yet.

        Todo:
            - Phase 1: Implement using cv2.perspectiveTransform (batched).

        """
        if not self.is_calibrated:
            raise RuntimeError("Homography not computed. Call compute_homography() first.")

        # TODO(Phase 1): points = image_points.reshape(-1, 1, 2).astype(np.float64)
        # TODO(Phase 1): court = cv2.perspectiveTransform(points, self._homography)
        # TODO(Phase 1): return court.reshape(-1, 2)
        logger.warning("project_points() not yet implemented (Phase 1 TODO)")
        return np.zeros_like(image_points, dtype=np.float64)

    def project_bounding_box_bottom(self, bbox: NDArray[np.float64]) -> CourtPoint:
        """Project the bottom-center of a bounding box to court-space.

        Players are assumed to stand on the court plane, so the bottom of
        their bounding box (feet position) provides the most accurate
        projection point.

        Args:
            bbox: Array of shape (4,) containing (x1, y1, x2, y2) in pixels.

        Returns:
            Array of shape (2,) with court-space coordinates (x, y) in meters.

        Todo:
            - Phase 1: Extract bottom-center: (x1+x2)/2, y2
            - Phase 1: Call self.project_point().

        """
        # TODO(Phase 1): x1, y1, x2, y2 = bbox
        # TODO(Phase 1): bottom_center = np.array([(x1 + x2) / 2, y2], dtype=np.float64)
        # TODO(Phase 1): return self.project_point(bottom_center)
        logger.warning("project_bounding_box_bottom() not yet implemented (Phase 1 TODO)")
        return np.zeros(2, dtype=np.float64)
