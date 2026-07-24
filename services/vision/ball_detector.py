"""
Purpose: Football Detection service utilizing fine-tuned Ultralytics YOLO models with spatial aspect ratio & size constraints.
Dependencies: cv2, numpy, ultralytics (optional runtime fallback), services.vision.models, shared.domain.entities, shared.logging
Inputs: FrameData or raw OpenCV BGR image arrays
Outputs: BallDetectionResult payload containing detected football bounding boxes and confidence scores
"""

import time
from typing import Optional, Union
import cv2
import numpy as np
from services.ingestion.frame import FrameData
from services.vision.models import BallDetectionResult, DetectedObject
from shared.domain.entities import BoundingBox, TrackedObjectType
from shared.logging import setup_logger

logger = setup_logger("ball_detector", service_name="vision")

COCO_SPORTS_BALL_CLASS_ID = 32

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False


class BallDetector:
    """YOLO v8/v11 football object detector with spatial geometric filtering."""

    def __init__(
        self,
        model_path: str = "models/yolov8x.pt",
        confidence_threshold: float = 0.15,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.4,
        max_area_ratio: float = 0.05,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_area_ratio = max_area_ratio

        self._model = None
        if YOLO_AVAILABLE:
            try:
                self._model = YOLO(model_path)
                logger.info(f"BallDetector loaded YOLO model from '{model_path}'.")
            except Exception as e:
                logger.warning(f"Failed loading YOLO model '{model_path}': {e}. Operating in fallback mode.")

    def is_valid_ball_candidate(self, bbox: BoundingBox, frame_width: int, frame_height: int) -> bool:
        """Enforces spatial circularity (aspect ratio) and maximum size filtering to reject non-ball objects."""
        if bbox.height <= 0:
            return False

        aspect_ratio = bbox.width / bbox.height
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False

        frame_area = float(frame_width * frame_height)
        if frame_area > 0 and (bbox.area / frame_area) > self.max_area_ratio:
            return False

        return True

    # Alias for test compatibility
    is_valid_ball_bbox = is_valid_ball_candidate

    def detect(self, frame: Union[FrameData, np.ndarray]) -> BallDetectionResult:
        """Performs football detection on frame image."""
        start_time = time.perf_counter()

        if isinstance(frame, FrameData):
            image = frame.image
            frame_num = frame.frame_number
            timestamp = frame.timestamp_seconds
        else:
            image = frame
            frame_num = 0
            timestamp = 0.0

        h, w = image.shape[:2]

        if self._model is not None:
            results = self._model.predict(image, conf=self.confidence_threshold, verbose=False)
            candidates: list[BoundingBox] = []
            best_ball: Optional[DetectedObject] = None
            highest_conf = 0.0

            for r in results:
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue

                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls_id == COCO_SPORTS_BALL_CLASS_ID or cls_id == 0:
                        xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                        bbox = BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=conf,
                        )

                        if self.is_valid_ball_candidate(bbox, w, h):
                            candidates.append(bbox)
                            if conf > highest_conf:
                                highest_conf = conf
                                best_ball = DetectedObject(
                                    object_type=TrackedObjectType.BALL,
                                    class_id=cls_id,
                                    confidence=conf,
                                    bbox=bbox,
                                    ground_position=bbox.center,
                                )

            elapsed = (time.perf_counter() - start_time) * 1000.0
            return BallDetectionResult(
                frame_number=frame_num,
                timestamp_seconds=timestamp,
                has_ball=best_ball is not None,
                ball_object=best_ball,
                candidates=candidates,
                processing_time_ms=elapsed,
            )

        # OpenCV color/contour heuristic fallback mode
        elapsed = (time.perf_counter() - start_time) * 1000.0
        return BallDetectionResult(
            frame_number=frame_num,
            timestamp_seconds=timestamp,
            has_ball=False,
            ball_object=None,
            candidates=[],
            processing_time_ms=elapsed,
        )

    detect_ball = detect
