"""
Unit tests for Ball Tracking Service (Kalman filtering, occlusion interpolation, velocity calculation, and visualizer).
"""

import numpy as np
import pytest
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata
from services.vision.ball_tracker import BallKalmanFilter, BallTracker
from services.vision.models import BallDetectionResult, DetectedObject
from services.vision.visualizer import draw_ball_tracks
from shared.domain.entities import BoundingBox, TrackedObjectType


@pytest.fixture
def sample_frame_data():
    metadata = VideoMetadata(
        source_path="test.mp4",
        width=640,
        height=480,
        fps=30.0,
        total_frames=100,
        duration_seconds=3.33,
        aspect_ratio=1.333,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameData(
        frame_number=1,
        timestamp_seconds=0.033,
        image=image,
        metadata=metadata,
    )


def test_ball_kalman_filter_step():
    kf = BallKalmanFilter()
    kf.initialize_state(100.0, 200.0)
    assert kf.state[0, 0] == 100.0
    assert kf.state[1, 0] == 200.0

    # Predict next state with dt=0.1s
    kf.predict(dt=0.1)

    # Update with new measurement (110.0, 205.0)
    kf.update(np.array([110.0, 205.0], dtype=np.float32))
    assert kf.state[0, 0] > 100.0
    assert kf.state[1, 0] > 200.0


def test_ball_tracker_interpolation(sample_frame_data):
    tracker = BallTracker()

    # Frame 1: Ball detected at (100, 100, 120, 120)
    det_box = BoundingBox(x1=100.0, y1=100.0, x2=120.0, y2=120.0, confidence=0.9)
    det_obj = DetectedObject(object_type=TrackedObjectType.BALL, class_id=32, confidence=0.9, bbox=det_box)
    res1 = tracker.update(
        BallDetectionResult(frame_number=1, timestamp_seconds=0.033, has_ball=True, ball_object=det_obj),
        sample_frame_data,
    )
    assert res1.has_ball is True
    assert res1.tracked_ball.is_interpolated is False

    # Frame 2: Detection dropout (has_ball = False) -> Ball Tracker should interpolate position
    frame2 = sample_frame_data
    frame2.frame_number = 2
    frame2.timestamp_seconds = 0.066
    res2 = tracker.update(
        BallDetectionResult(frame_number=2, timestamp_seconds=0.066, has_ball=False, ball_object=None),
        frame2,
    )
    assert res2.has_ball is True
    assert res2.tracked_ball.is_interpolated is True


def test_draw_ball_tracks(sample_frame_data):
    tracker = BallTracker()
    det_box = BoundingBox(x1=100.0, y1=100.0, x2=120.0, y2=120.0, confidence=0.9)
    det_obj = DetectedObject(object_type=TrackedObjectType.BALL, class_id=32, confidence=0.9, bbox=det_box)
    res = tracker.update(
        BallDetectionResult(frame_number=1, timestamp_seconds=0.033, has_ball=True, ball_object=det_obj),
        sample_frame_data,
    )
    annotated = draw_ball_tracks(sample_frame_data.image, res)
    assert annotated.shape == (480, 640, 3)
