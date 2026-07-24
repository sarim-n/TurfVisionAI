"""
Unit tests for Football Detection Service models, spatial filtering, and visualizer.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata
from services.vision.ball_detector import COCO_SPORTS_BALL_CLASS_ID, BallDetector
from services.vision.models import BallDetectionResult, DetectedObject
from services.vision.visualizer import draw_ball_detections
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


def test_spatial_aspect_ratio_filter():
    detector = BallDetector(min_aspect_ratio=0.6, max_aspect_ratio=1.4, max_area_ratio=0.05)

    # Valid square ball box (20x20 pixels)
    valid_box = BoundingBox(x1=100.0, y1=100.0, x2=120.0, y2=120.0, confidence=0.85)
    assert detector.is_valid_ball_bbox(valid_box, 640, 480) is True

    # Invalid stretched rectangular box (100x10 pixels -> aspect ratio 10.0)
    wide_box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=110.0, confidence=0.85)
    assert detector.is_valid_ball_bbox(wide_box, 640, 480) is False

    # Invalid giant box covering > 5% of frame
    huge_box = BoundingBox(x1=0.0, y1=0.0, x2=400.0, y2=400.0, confidence=0.85)
    assert detector.is_valid_ball_bbox(huge_box, 640, 480) is False


def test_draw_ball_detections(sample_frame_data):
    bbox = BoundingBox(x1=100.0, y1=100.0, x2=120.0, y2=120.0, confidence=0.90)
    obj = DetectedObject(
        object_type=TrackedObjectType.BALL,
        class_id=32,
        confidence=0.90,
        bbox=bbox,
    )
    res = BallDetectionResult(
        frame_number=1,
        timestamp_seconds=0.033,
        has_ball=True,
        ball_object=obj,
    )
    annotated = draw_ball_detections(sample_frame_data.image, res)
    assert annotated.shape == (480, 640, 3)


@patch("services.vision.ball_detector.YOLO", create=True)
def test_ball_detector_detect(mock_yolo_cls, sample_frame_data):
    # Mock Ultralytics YOLO output for ball
    mock_box = MagicMock()
    mock_box.xyxy = [np.array([200.0, 200.0, 220.0, 220.0])]
    mock_box.conf = [0.85]
    mock_box.cls = [COCO_SPORTS_BALL_CLASS_ID]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = [mock_result]

    detector = BallDetector(model_path="dummy_ball_yolo.pt", confidence_threshold=0.2)
    detector._model = mock_model_instance

    res = detector.detect_ball(sample_frame_data)
    assert res.has_ball is True
    assert res.ball_object is not None
    assert res.ball_object.confidence == 0.85
    assert res.ball_object.bbox.center.x == 210.0
