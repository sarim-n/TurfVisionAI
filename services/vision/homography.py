"""
Purpose: Perspective transformation & pitch calibration engine using OpenCV homography matrices.
Dependencies: cv2, numpy, services.vision.models, shared.domain.entities, shared.logging
Inputs: 2D pixel coordinates, image dimension bounds
Outputs: 2D pitch coordinates in real-world meters (0..105m, 0..68m)
"""

from typing import Optional, Union
import cv2
import numpy as np
from services.vision.models import PitchDimensions
from shared.domain.entities import Point2D
from shared.logging import setup_logger

logger = setup_logger("pitch_homography", service_name="vision")


class PitchHomography:
    """Computes and applies 3x3 homography transformation between camera pixel coordinates and pitch meters."""

    def __init__(
        self,
        homography_matrix_or_src: Union[np.ndarray, list[tuple[float, float]]],
        dst_points: Optional[list[tuple[float, float]]] = None,
        dimensions: Optional[PitchDimensions] = None,
    ):
        if isinstance(homography_matrix_or_src, list) and dst_points is not None:
            src_pts = np.array(homography_matrix_or_src, dtype=np.float32)
            dst_pts = np.array(dst_points, dtype=np.float32)
            H, _ = cv2.findHomography(src_pts, dst_pts)
            homography_matrix = H
        else:
            homography_matrix = homography_matrix_or_src

        if homography_matrix is None or homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3 float array")

        self.H = homography_matrix.astype(np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.dimensions = dimensions or PitchDimensions()

    @classmethod
    def create_default_calibration(cls, image_width: int, image_height: int) -> "PitchHomography":
        """Generates synthetic 4-point corner calibration for fallback pitch mapping."""
        src_points = np.array(
            [
                [0.0, 0.0],
                [float(image_width), 0.0],
                [float(image_width), float(image_height)],
                [0.0, float(image_height)],
            ],
            dtype=np.float32,
        )

        # Standard Pitch bounds: 105m x 68m
        dst_points = np.array(
            [
                [0.0, 0.0],
                [105.0, 0.0],
                [105.0, 68.0],
                [0.0, 68.0],
            ],
            dtype=np.float32,
        )

        H, _ = cv2.findHomography(src_points, dst_points)
        logger.info("Computed 3x3 Pitch Homography matrix successfully.")
        return cls(H)

    def pixel_to_pitch(self, point: Point2D) -> Point2D:
        """Transforms 2D pixel canvas coordinate to 2D pitch coordinate in meters."""
        vec = np.array([point.x, point.y, 1.0], dtype=np.float64)
        transformed = np.dot(self.H, vec)
        if abs(transformed[2]) < 1e-6:
            return Point2D(x=0.0, y=0.0)
        x_m = transformed[0] / transformed[2]
        y_m = transformed[1] / transformed[2]
        return Point2D(x=float(x_m), y=float(y_m))

    # Alias for visualizer and test suite
    transform_pixel_to_pitch = pixel_to_pitch

    def pitch_to_pixel(self, point: Point2D) -> Point2D:
        """Transforms 2D pitch coordinate in meters back to 2D pixel canvas coordinate."""
        vec = np.array([point.x, point.y, 1.0], dtype=np.float64)
        transformed = np.dot(self.H_inv, vec)
        if abs(transformed[2]) < 1e-6:
            return Point2D(x=0.0, y=0.0)
        px = transformed[0] / transformed[2]
        py = transformed[1] / transformed[2]
        return Point2D(x=float(px), y=float(py))

    transform_pitch_to_pixel = pitch_to_pixel
