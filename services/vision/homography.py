"""
Purpose: Camera Calibration & Perspective Transformation (Homography) Engine.
Dependencies: cv2, numpy, services.vision.models, shared.domain.entities, shared.logging
Inputs: Pixel coordinates [(x,y)] and corresponding real-world pitch metric coordinates [(X,Y)]
Outputs: PitchHomography instance transforming pixel points to real-world meters
"""

from typing import Sequence
import cv2
import numpy as np
from services.vision.models import PitchDimensions, PitchPoint
from shared.domain.entities import Point2D
from shared.logging import setup_logger

logger = setup_logger("pitch_homography", service_name="vision")


class PitchHomography:
    """Computes and applies 3x3 Homography matrix transformations between camera pixel space and 2D pitch metric space."""

    def __init__(
        self,
        pixel_points: Sequence[tuple[float, float]],
        pitch_points: Sequence[tuple[float, float]],
        pitch_dimensions: PitchDimensions = PitchDimensions(),
    ):
        if len(pixel_points) < 4 or len(pitch_points) < 4 or len(pixel_points) != len(pitch_points):
            raise ValueError("Homography calibration requires at least 4 matching point pairs.")

        self.pitch_dimensions = pitch_dimensions
        self.pixel_points = np.array(pixel_points, dtype=np.float32)
        self.pitch_points = np.array(pitch_points, dtype=np.float32)

        # Compute 3x3 Homography matrix H (pixel -> pitch)
        self.H, status = cv2.findHomography(self.pixel_points, self.pitch_points, cv2.RANSAC, 5.0)
        if self.H is None:
            raise ValueError("Failed to compute Homography matrix from provided point correspondences.")

        # Compute Inverse Homography matrix H_inv (pitch -> pixel)
        self.H_inv = np.linalg.inv(self.H)
        logger.info("Computed 3x3 Pitch Homography matrix successfully.")

    def pixel_to_pitch(self, pixel_pt: Point2D) -> PitchPoint:
        """Transforms a 2D pixel point (x, y) into real-world 2D pitch coordinates (meters)."""
        src = np.array([[[pixel_pt.x, pixel_pt.y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H)
        x_meters = float(dst[0, 0, 0])
        y_meters = float(dst[0, 0, 1])

        # Clamp to pitch dimensions
        x_clamped = max(0.0, min(self.pitch_dimensions.length_meters, x_meters))
        y_clamped = max(0.0, min(self.pitch_dimensions.width_meters, y_meters))

        return PitchPoint(x_meters=round(x_clamped, 2), y_meters=round(y_clamped, 2))

    def pitch_to_pixel(self, pitch_pt: PitchPoint) -> Point2D:
        """Transforms real-world 2D pitch coordinates (meters) into camera pixel coordinates (x, y)."""
        src = np.array([[[pitch_pt.x_meters, pitch_pt.y_meters]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.H_inv)
        px = float(dst[0, 0, 0])
        py = float(dst[0, 0, 1])
        return Point2D(x=round(px, 1), y=round(py, 1))

    @classmethod
    def create_default_calibration(
        cls,
        image_width: int,
        image_height: int,
        pitch_dimensions: PitchDimensions = PitchDimensions(),
    ) -> "PitchHomography":
        """Creates a standard synthetic bounding box perspective mapping for full-frame camera feeds."""
        # 4 corners of frame -> 4 corners of pitch
        pixel_corners = [
            (0.0, 0.0),
            (float(image_width), 0.0),
            (float(image_width), float(image_height)),
            (0.0, float(image_height)),
        ]
        pitch_corners = [
            (0.0, 0.0),
            (pitch_dimensions.length_meters, 0.0),
            (pitch_dimensions.length_meters, pitch_dimensions.width_meters),
            (0.0, pitch_dimensions.width_meters),
        ]
        return cls(pixel_corners, pitch_corners, pitch_dimensions)
