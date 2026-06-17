"""Tests for the CourtProjector homography module.

Phase 0: Only structural validation (class instantiation, property defaults).
Phase 1 will add full numerical tests once compute_homography is implemented.
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
        """The expected number of manual keypoints should be 12."""
        assert CourtProjector.NUM_KEYPOINTS == 12


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
