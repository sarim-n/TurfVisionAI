"""
Purpose: Player Detection Engine using Ultralytics YOLOv8 for COCO person class 0.
Dependencies: torch, ultralytics, cv2, numpy, services.vision.models, shared.domain.entities, shared.logging
Inputs: Raw OpenCV BGR image arrays or FrameData objects
Outputs: PlayerDetectionResult domain objects containing normalized BoundingBox lists
"""

import time
from typing import Optional, Union
import numpy as np
from services.ingestion.frame import FrameData
from services.vision.models import DetectedObject, PlayerDetectionResult
from shared.domain.entities import BoundingBox, Point2D, TrackedObjectType
from shared.logging import setup_logger

logger = setup_logger("player_detector", service_name="vision")

COCO_PERSON_CLASS_ID = 0


class PlayerDetector:
    """YOLOv8 Player Detector optimized for football pitch person detection (COCO Class 0)."""

    def __init__(
        self,
        model_path: str = "models/yolov8x.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        """Lazy loader for Ultralytics YOLO model with fallback handling."""
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore

                logger.info(f"Loading YOLO model from '{self.model_path}' on device '{self.device}'...")
                self._model = YOLO(self.model_path)
                logger.info("YOLO model loaded successfully.")
            except Exception as e:
                logger.warning(f"Ultralytics YOLO unavailable or model missing ('{self.model_path}'): {e}. Operating in fallback mode.")
                self._model = "fallback"

    def detect(self, frame_data: Union[FrameData, np.ndarray], confidence_threshold: Optional[float] = None) -> PlayerDetectionResult:
        """Runs player detection on a single frame."""
        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        img = frame_data.image if isinstance(frame_data, FrameData) else frame_data
        frame_num = frame_data.frame_number if isinstance(frame_data, FrameData) else 0
        ts = frame_data.timestamp_seconds if isinstance(frame_data, FrameData) else 0.0

        results = self.detect_batch([img], confidence_threshold=conf)
        res = results[0]
        res.frame_number = frame_num
        res.timestamp_seconds = ts
        return res

    def detect_batch(
        self, frames: list[np.ndarray], confidence_threshold: Optional[float] = None
    ) -> list[PlayerDetectionResult]:
        """Runs batch inference on a list of frames."""
        self._load_model()
        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        results: list[PlayerDetectionResult] = []

        if self._model == "fallback" or self._model is None:
            for _ in frames:
                results.append(
                    PlayerDetectionResult(
                        frame_number=0,
                        timestamp_seconds=0.0,
                        detections=[],
                        processing_time_ms=0.0,
                    )
                )
            return results

        start_time = time.perf_counter()

        for idx, frame in enumerate(frames):
            frame_height, frame_width = frame.shape[:2]
            detections: list[DetectedObject] = []

            # YOLO inference (COCO class 0 = person)
            predictions = self._model.predict(
                frame,
                classes=[COCO_PERSON_CLASS_ID],
                conf=conf,
                device=self.device,
                verbose=False,
            )

            if len(predictions) > 0 and predictions[0].boxes is not None:
                boxes = predictions[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf_val = float(box.conf[0])

                    bbox = BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=conf_val,
                    )
                    # Ground feet contact point (center bottom of bounding box)
                    feet_x = (x1 + x2) / 2.0
                    feet_y = y2

                    obj = DetectedObject(
                        object_type=TrackedObjectType.PLAYER,
                        class_id=COCO_PERSON_CLASS_ID,
                        confidence=conf_val,
                        bbox=bbox,
                        ground_position=Point2D(x=feet_x, y=feet_y),
                    )
                    detections.append(obj)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / len(frames)
            results.append(
                PlayerDetectionResult(
                    frame_number=idx,
                    timestamp_seconds=0.0,
                    detections=detections,
                    processing_time_ms=elapsed_ms,
                )
            )

        return results
