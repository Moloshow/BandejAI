"""Homography-based court projection.

Transforms image coordinates (u, v) to 2D court-space (x, y) using a
perspective homography matrix computed from court keypoints.

This is a deterministic, explainable module - no neural networks involved.
The homography matrix H is computed once from manually-annotated keypoints
(see main.py: run_homography_init) and then applied to every frame.

Mathematical formulation:
    [[x], [y], [1]] = c * H * [[u], [v], [1]]

where c is a scaling factor to normalize the homogeneous coordinate back to 1.

Status: Phase 1A (implemented & tested).
"""

from __future__ import annotations

import logging
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("bandejai.homography")

# Type aliases for clarity (float64 arrays)
ImagePoint = NDArray[np.float64]  # shape (2,) or (N, 2)
CourtPoint = NDArray[np.float64]  # shape (2,) or (N, 2)


class CourtProjector:
    """Projects image-space pixel coordinates onto a 2D padel court model.

    The court model uses a coordinate system in meters, with the origin at
    the **left corner of the near baseline**. The x-axis runs along the
    baseline (0 to COURT_WIDTH_M), the y-axis runs toward the far baseline
    (0 to COURT_LENGTH_M).

    Coordinate system (bird's-eye view, full court)::

        x=0   x=5   x=10
         |     |     |
        [12]--[13]-[14]  y=20    (far baseline)
         |     |     |
         [9]--[10]-[11]  y=16.95 (far service line)
         |     |     |
         [6]--[ 7]-[ 8]  y=10    (net)
         |     |     |
         [3]--[ 4]-[ 5]  y=3.05  (near service line)
         |     |     |
         [0]--[ 1]-[ 2]  y=0     (near baseline)

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

    # Service line distances from each baseline (FIP regulations), in meters
    # The service box is 6.95m from the net, so from the baseline it's 10 - 6.95 = 3.05m
    SERVICE_LINE_DIST_M: ClassVar[float] = 3.05

    # Expected number of keypoints for manual annotation (5 rows x 3 columns)
    NUM_KEYPOINTS: ClassVar[int] = 15

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

    @property
    def court_keypoints_template(self) -> NDArray[np.float64]:
        """Return the 15 standard court keypoints in meters (court-space).

        These are the real-world coordinates (x, y) of the 15 line
        intersections used for manual annotation. The user must click the
        corresponding pixel positions in the image.

        Returns:
            Array of shape (15, 2) with court-space coordinates in meters.

        Keypoint indices (see class docstring for visual layout)::

              0: Near baseline left      ( 0.0,  0.00)
              1: Near baseline center    ( 5.0,  0.00)
              2: Near baseline right    (10.0,  0.00)
              3: Near service line left  ( 0.0,  3.05)
              4: Near service line center( 5.0,  3.05)
              5: Near service line right (10.0,  3.05)
              6: Net left                ( 0.0, 10.00)
              7: Net center              ( 5.0, 10.00)
              8: Net right              (10.0, 10.00)
              9: Far service line left   ( 0.0, 16.95)
             10: Far service line center ( 5.0, 16.95)
             11: Far service line right  (10.0, 16.95)
             12: Far baseline left       ( 0.0, 20.00)
             13: Far baseline center     ( 5.0, 20.00)
             14: Far baseline right     (10.0, 20.00)

        """
        w = self.COURT_WIDTH_M
        length = self.COURT_LENGTH_M
        s = self.SERVICE_LINE_DIST_M
        return np.array(
            [
                [0.0, 0.0],  # 0: Near baseline left
                [w / 2, 0.0],  # 1: Near baseline center
                [w, 0.0],  # 2: Near baseline right
                [0.0, s],  # 3: Near service line left
                [w / 2, s],  # 4: Near service line center
                [w, s],  # 5: Near service line right
                [0.0, length / 2],  # 6: Net left
                [w / 2, length / 2],  # 7: Net center
                [w, length / 2],  # 8: Net right
                [0.0, length - s],  # 9: Far service line left
                [w / 2, length - s],  # 10: Far service line center
                [w, length - s],  # 11: Far service line right
                [0.0, length],  # 12: Far baseline left
                [w / 2, length],  # 13: Far baseline center
                [w, length],  # 14: Far baseline right
            ],
        )

    def get_template(self, mode: int = 15) -> NDArray[np.float64]:
        """Get the court keypoints template for a specific configuration mode.

        Args:
            mode: Number of keypoints (15=full, 12=intersections, 10=edges, 6=corners+net, 4=corners).

        Returns:
            Array of shape (mode, 2) containing the court coordinates in meters.
        """
        if mode == 15:
            indices = list(range(15))
        elif mode == 12:
            indices = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14]
        elif mode == 10:
            indices = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14]
        elif mode == 6:
            indices = [0, 2, 6, 8, 12, 14]
        elif mode == 4:
            indices = [0, 2, 12, 14]
        else:
            raise ValueError(f"Unsupported points mode: {mode}")

        return self.court_keypoints_template[indices]

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

        # Convert to float32 for OpenCV
        img_pts = np.asarray(image_points, dtype=np.float32)
        crt_pts = np.asarray(court_points, dtype=np.float32)

        # Compute homography with RANSAC.
        # We compute H_inv (court -> image) first to minimize error in pixel space,
        # which is essential for accurate visual projection at the near baseline.
        # Here, the threshold 5.0 is correctly interpreted as 5.0 pixels.
        H_inv, mask = cv2.findHomography(crt_pts, img_pts, cv2.RANSAC, 5.0)

        if H_inv is None:
            raise RuntimeError(
                "cv2.findHomography failed to compute a valid matrix. "
                "Check that the point correspondences are non-degenerate "
                "(no 3 points collinear)."
            )

        self._homography = np.linalg.inv(H_inv).astype(np.float64)

        inliers = int(mask.sum()) if mask is not None else len(img_pts)
        logger.info(
            "Homography computed: %d/%d points are inliers",
            inliers,
            len(img_pts),
        )

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

        """
        if not self.is_calibrated or self._homography is None:
            raise RuntimeError("Homography not computed. Call compute_homography() first.")

        H = self._homography  # type narrowing for mypy/pyright
        point = np.array([[image_point]], dtype=np.float32)
        court = cv2.perspectiveTransform(point, H)
        return court[0, 0].astype(np.float64)

    def project_points(self, image_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project multiple image-space points to 2D court-space.

        Args:
            image_points: Array of shape (N, 2) with pixel coordinates.

        Returns:
            Array of shape (N, 2) with court-space coordinates in meters.

        Raises:
            RuntimeError: If the homography has not been computed yet.

        """
        if not self.is_calibrated or self._homography is None:
            raise RuntimeError("Homography not computed. Call compute_homography() first.")

        H = self._homography  # type narrowing for mypy/pyright
        points = np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2)
        court = cv2.perspectiveTransform(points, H)
        return court.reshape(-1, 2).astype(np.float64)

    def project_bounding_box_bottom(self, bbox: NDArray[np.float64]) -> CourtPoint:
        """Project the bottom-center of a bounding box to court-space.

        Players are assumed to stand on the court plane, so the bottom of
        their bounding box (feet position) provides the most accurate
        projection point.

        Args:
            bbox: Array of shape (4,) containing (x1, y1, x2, y2) in pixels.

        Returns:
            Array of shape (2,) with court-space coordinates (x, y) in meters.

        """
        x1, y1, x2, y2 = bbox
        bottom_center = np.array([(x1 + x2) / 2, y2], dtype=np.float64)
        return self.project_point(bottom_center)
