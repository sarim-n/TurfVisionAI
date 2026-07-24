"""
Unit tests for Player Tracking Service (IoU tracking, track ID persistence, trajectory history, and visualizer).
"""

import numpy as np
import pytest
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata
from services.vision.models import DetectedObject, PlayerDetectionResult
from services.vision.player_tracker import PlayerTracker, compute_iou
from services.vision.visualizer import draw_player_tracks
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


def test_compute_iou():
    box1 = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0, confidence=1.0)
    box2 = BoundingBox(x1=5.0, y1=0.0, x2=15.0, y2=10.0, confidence=1.0)
    # Intersection = 5x10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 50/150 = 0.333
    iou = compute_iou(box1, box2)
    assert abs(iou - 0.333) < 0.01


def test_player_tracker_id_persistence(sample_frame_data):
    tracker = PlayerTracker(max_history_len=10)

    # Frame 1: Player at (10, 10, 50, 100)
    det1 = DetectedObject(
        object_type=TrackedObjectType.PLAYER,
        class_id=0,
        confidence=0.9,
        bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=100.0, confidence=0.9),
    )
    res1 = tracker.update(
        PlayerDetectionResult(frame_number=1, timestamp_seconds=0.033, detections=[det1]),
        sample_frame_data,
    )
    assert res1.active_player_count == 1
    p1_id = res1.tracked_players[0].track_id

    # Frame 2: Player moves slightly to (12, 12, 52, 102) -> IoU > 0.3
    det2 = DetectedObject(
        object_type=TrackedObjectType.PLAYER,
        class_id=0,
        confidence=0.92,
        bbox=BoundingBox(x1=12.0, y1=12.0, x2=52.0, y2=102.0, confidence=0.92),
    )
    frame2 = sample_frame_data
    frame2.frame_number = 2
    frame2.timestamp_seconds = 0.066

    res2 = tracker.update(
        PlayerDetectionResult(frame_number=2, timestamp_seconds=0.066, detections=[det2]),
        frame2,
    )
    assert res2.active_player_count == 1
    # Check ID persistence
    assert res2.tracked_players[0].track_id == p1_id
    assert len(res2.tracked_players[0].trajectory_history) == 2


def test_draw_player_tracks(sample_frame_data):
    det = DetectedObject(
        object_type=TrackedObjectType.PLAYER,
        class_id=0,
        confidence=0.9,
        bbox=BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=100.0, confidence=0.9),
    )
    tracker = PlayerTracker()
    res = tracker.update(
        PlayerDetectionResult(frame_number=1, timestamp_seconds=0.033, detections=[det]),
        sample_frame_data,
    )
    annotated = draw_player_tracks(sample_frame_data.image, res)
    assert annotated.shape == (480, 640, 3)
