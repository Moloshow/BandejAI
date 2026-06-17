"""Tests for the CourtProjector homography module.

Phase 1: Full numerical tests (homography computation, reprojection accuracy,
keypoints template, bounding box projection).
"""

from __future__ import annotations

import numpy as np
import pytest

from core_math.homography.projector import CourtProjector


class TestCourtProjectorInit:
    """Test default state of CourtProjector."""

    def test_projector_starts_uncalibrated(self) -> None:
        """A new projector should not be calibrated."""
        projector = CourtProjector()
        assert not projector.is_calibrated

    def test_homography_is_none_by_default(self) -> None:
        """The homography matrix should be None before calibration."""
        projector = CourtProjector()
        assert projector.homography is None

    def test_court_dimensions_match_fip(self) -> None:
        """Court dimensions should match FIP regulations (10m x 20m)."""
        assert CourtProjector.COURT_WIDTH_M == 10.0
        assert CourtProjector.COURT_LENGTH_M == 20.0
        assert CourtProjector.NET_HEIGHT_M == 0.88

    def test_num_keypoints(self) -> None:
        """The expected number of manual keypoints should be 15."""
        assert CourtProjector.NUM_KEYPOINTS == 15


class TestCourtProjectorValidation:
    """Test input validation for compute_homography (raises before computation)."""

    def test_compute_rejects_too_few_points(self) -> None:
        """Fewer than 4 point pairs should raise ValueError."""
        projector = CourtProjector()
        with pytest.raises(ValueError, match="At least 4"):
            projector.compute_homography(
                image_points=np.zeros((3, 2)),
                court_points=np.zeros((3, 2)),
            )

    def test_compute_rejects_shape_mismatch(self) -> None:
        """Mismatched image/court shapes should raise ValueError."""
        projector = CourtProjector()
        with pytest.raises(ValueError, match="Shape mismatch"):
            projector.compute_homography(
                image_points=np.zeros((4, 2)),
                court_points=np.zeros((5, 2)),
            )

    def test_project_point_raises_when_uncalibrated(self) -> None:
        """Projecting before calibration should raise RuntimeError."""
        projector = CourtProjector()
        with pytest.raises(RuntimeError, match="not computed"):
            projector.project_point(np.array([100.0, 200.0]))


class TestCourtKeypointsTemplate:
    """Test the 15-keypoint court template."""

    def test_template_has_15_keypoints(self) -> None:
        """Template should return exactly 15 keypoints."""
        projector = CourtProjector()
        template = projector.court_keypoints_template
        assert template.shape == (15, 2)

    def test_template_baselines(self) -> None:
        """Baseline keypoints should be at y=0."""
        projector = CourtProjector()
        template = projector.court_keypoints_template
        # Keypoints 0, 1, 2 are on the baseline
        assert template[0, 1] == 0.0  # Baseline left
        assert template[1, 1] == 0.0  # Baseline right
        assert template[2, 1] == 0.0  # Baseline center
        # x coordinates
        assert template[0, 0] == 0.0
        assert template[1, 0] == 5.0
        assert template[2, 0] == 10.0

    def test_template_net_at_midpoint(self) -> None:
        """Net keypoints should be at y=10 (midpoint of 20m court)."""
        projector = CourtProjector()
        template = projector.court_keypoints_template
        # Keypoints 6, 7, 8 are on the net
        assert template[6, 1] == 10.0
        assert template[7, 1] == 10.0
        assert template[8, 1] == 10.0

    def test_template_service_lines(self) -> None:
        """Service lines should be at y=3.05 and y=16.95."""
        projector = CourtProjector()
        template = projector.court_keypoints_template
        # Near service line (y=3.05)
        assert template[3, 1] == pytest.approx(3.05)
        assert template[5, 1] == pytest.approx(3.05)
        # Far service line (y=16.95)
        assert template[9, 1] == pytest.approx(16.95)
        assert template[11, 1] == pytest.approx(16.95)

    def test_template_far_baseline(self) -> None:
        """Far baseline keypoints should be at y=20 (COURT_LENGTH_M)."""
        projector = CourtProjector()
        template = projector.court_keypoints_template
        # Keypoints 12, 13, 14 are on the far baseline
        assert template[12, 1] == pytest.approx(20.0)
        assert template[13, 1] == pytest.approx(20.0)
        assert template[14, 1] == pytest.approx(20.0)
        # x coordinates
        assert template[12, 0] == pytest.approx(0.0)
        assert template[13, 0] == pytest.approx(5.0)
        assert template[14, 0] == pytest.approx(10.0)


class TestHomographyComputation:
    """Test homography computation and reprojection accuracy."""

    def test_is_calibrated_after_compute(self) -> None:
        """Projector should be calibrated after successful compute."""
        projector = CourtProjector()
        # Use a simple affine transform as ground truth
        image_pts = np.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=np.float64)
        court_pts = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float64)
        projector.compute_homography(image_pts, court_pts)
        assert projector.is_calibrated
        assert projector.homography is not None
        assert projector.homography.shape == (3, 3)

    def test_reprojection_accuracy_identity(self) -> None:
        """Reprojecting calibration points should give near-exact results."""
        projector = CourtProjector()
        image_pts = np.array(
            [[0, 0], [100, 0], [100, 200], [0, 200], [50, 100]], dtype=np.float64
        )
        court_pts = np.array(
            [[0, 0], [10, 0], [10, 20], [0, 20], [5, 10]], dtype=np.float64
        )
        projector.compute_homography(image_pts, court_pts)

        # Reproject and check accuracy
        projected = projector.project_points(image_pts)
        errors = np.linalg.norm(projected - court_pts, axis=1)
        assert np.max(errors) < 0.01  # Less than 1cm error

    def test_project_single_point(self) -> None:
        """project_point should return a single (x, y) coordinate."""
        projector = CourtProjector()
        image_pts = np.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=np.float64)
        court_pts = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float64)
        projector.compute_homography(image_pts, court_pts)

        result = projector.project_point(np.array([50.0, 100.0]))
        assert result.shape == (2,)
        # Center of the court
        assert result[0] == pytest.approx(5.0, abs=0.01)
        assert result[1] == pytest.approx(10.0, abs=0.01)

    def test_project_bounding_box_bottom(self) -> None:
        """project_bounding_box_bottom should use the bottom-center of bbox."""
        projector = CourtProjector()
        # Square court: 100px = 10m, 200px = 20m
        image_pts = np.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=np.float64)
        court_pts = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float64)
        projector.compute_homography(image_pts, court_pts)

        # bbox at center-bottom of image: (40, 80) to (60, 100)
        # Bottom center = (50, 100) -> court (5, 10)
        result = projector.project_bounding_box_bottom(np.array([40.0, 80.0, 60.0, 100.0]))
        assert result[0] == pytest.approx(5.0, abs=0.01)
        assert result[1] == pytest.approx(10.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
