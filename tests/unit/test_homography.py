"""
Unit tests for Camera Calibration & Pitch Homography Engine.
"""

import numpy as np
import pytest
from services.vision.homography import PitchHomography
from services.vision.models import PitchDimensions, TrackedBall, TrackedPlayer
from services.vision.visualizer import draw_birds_eye_view
from shared.domain.entities import BoundingBox, Point2D


def test_pitch_homography_forward_inverse_transformation():
    # 4 synthetic image pixel points (640x480) -> 4 pitch corner meters (105x68)
    pixel_pts = [(0.0, 0.0), (640.0, 0.0), (640.0, 480.0), (0.0, 480.0)]
    pitch_pts = [(0.0, 0.0), (105.0, 0.0), (105.0, 68.0), (0.0, 68.0)]

    homography = PitchHomography(pixel_pts, pitch_pts)

    # Top-left corner (0,0) -> Pitch (0.0, 0.0)
    p0 = homography.pixel_to_pitch(Point2D(x=0.0, y=0.0))
    assert abs(p0.x_meters - 0.0) < 0.1
    assert abs(p0.y_meters - 0.0) < 0.1

    # Center of frame (320, 240) -> Center of pitch (52.5, 34.0)
    p_center = homography.pixel_to_pitch(Point2D(x=320.0, y=240.0))
    assert abs(p_center.x_meters - 52.5) < 0.5
    assert abs(p_center.y_meters - 34.0) < 0.5

    # Inverse transform center of pitch (52.5, 34.0) back to pixel (320.0, 240.0)
    pix_center = homography.pitch_to_pixel(p_center)
    assert abs(pix_center.x - 320.0) < 1.0
    assert abs(pix_center.y - 240.0) < 1.0


def test_create_default_calibration():
    homography = PitchHomography.create_default_calibration(1920, 1080)
    assert homography.H is not None
    p = homography.pixel_to_pitch(Point2D(x=960.0, y=540.0))
    assert abs(p.x_meters - 52.5) < 0.5


def test_draw_birds_eye_view():
    homography = PitchHomography.create_default_calibration(640, 480)
    player = TrackedPlayer(
        track_id=1,
        bbox=BoundingBox(x1=300.0, y1=200.0, x2=340.0, y2=280.0, confidence=0.9),
        ground_position=Point2D(x=320.0, y=280.0),
    )
    ball = TrackedBall(
        center=Point2D(x=330.0, y=285.0),
        velocity=Point2D(x=10.0, y=5.0),
        speed_px_per_sec=11.18,
    )
    canvas = draw_birds_eye_view([player], ball, homography, canvas_size=(600, 400))
    assert canvas.shape == (400, 600, 3)
