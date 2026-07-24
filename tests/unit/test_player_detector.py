"""
Unit tests for Player Detection Service models, detector interface, and visualizer.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata
from services.vision.models import DetectedObject, PlayerDetectionResult
from services.vision.player_detector import COCO_PERSON_CLASS_ID, PlayerDetector
from services.vision.visualizer import draw_player_detections
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


def test_player_detection_result_model():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=100.0, confidence=0.88)
    obj = DetectedObject(
        object_type=TrackedObjectType.PLAYER,
        class_id=0,
        confidence=0.88,
        bbox=bbox,
    )
    res = PlayerDetectionResult(
        frame_number=1,
        timestamp_seconds=0.033,
        detections=[obj],
        inference_time_ms=12.5,
    )
    assert res.player_count == 1
    assert res.detections[0].bbox.center.x == 30.0


def test_draw_player_detections(sample_frame_data):
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=100.0, confidence=0.90)
    obj = DetectedObject(
        object_type=TrackedObjectType.PLAYER,
        class_id=0,
        confidence=0.90,
        bbox=bbox,
    )
    res = PlayerDetectionResult(
        frame_number=1,
        timestamp_seconds=0.033,
        detections=[obj],
    )
    annotated = draw_player_detections(sample_frame_data.image, res)
    assert annotated.shape == (480, 640, 3)


@patch("services.vision.player_detector.YOLO", create=True)
def test_player_detector_detect(mock_yolo_cls, sample_frame_data):
    # Mock Ultralytics YOLO output box
    mock_box = MagicMock()
    mock_box.xyxy = [np.array([100.0, 150.0, 200.0, 350.0])]
    mock_box.conf = [0.92]
    mock_box.cls = [COCO_PERSON_CLASS_ID]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = [mock_result]

    detector = PlayerDetector(model_path="dummy_yolo.pt", confidence_threshold=0.5)
    detector._model = mock_model_instance

    res = detector.detect(sample_frame_data)
    assert res.player_count == 1
    assert res.detections[0].confidence == 0.92
    assert res.detections[0].bbox.x1 == 100.0
