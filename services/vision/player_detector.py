"""
Purpose: Player Detection Service using Ultralytics YOLO.
Dependencies: ultralytics, torch, time, services.ingestion.frame, shared.domain.entities
Inputs: FrameData objects
Outputs: PlayerDetectionResult domain objects containing player bounding boxes
"""

import time
from typing import Optional
from services.ingestion.frame import FrameData
from services.vision.models import DetectedObject, PlayerDetectionResult
from shared.config import get_settings
from shared.domain.entities import BoundingBox, TrackedObjectType
from shared.logging import setup_logger

logger = setup_logger("player_detector", service_name="vision")

# COCO class ID 0 is 'person'
COCO_PERSON_CLASS_ID = 0


class PlayerDetector:
    """YOLO-based Player Detector for identifying players and referees in video frames."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        settings = get_settings()
        self.model_path = model_path or settings.YOLO_MODEL_PATH
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.CONFIDENCE_THRESHOLD
        )
        self.device = device or settings.DEVICE
        self._model = None

    def _load_model(self) -> None:
        """Lazy loader for Ultralytics YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore

                logger.info(f"Loading YOLO model from '{self.model_path}' on device '{self.device}'...")
                self._model = YOLO(self.model_path)
                logger.info("YOLO model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load YOLO model from '{self.model_path}': {e}")
                raise RuntimeError(f"YOLO initialization error: {e}") from e

    def detect(self, frame_data: FrameData) -> PlayerDetectionResult:
        """Detects players in a single FrameData object."""
        results = self.detect_batch([frame_data])
        return results[0]

    def detect_batch(self, frames: list[FrameData]) -> list[PlayerDetectionResult]:
        """Performs batch detection over a list of FrameData objects."""
        if not frames:
            return []

        self._load_model()
        images = [f.image for f in frames]

        start_time = time.perf_counter()
        # Run YOLO inference
        raw_results = self._model.predict(
            source=images,
            conf=self.confidence_threshold,
            classes=[COCO_PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        per_frame_time_ms = total_time_ms / len(frames)

        detection_results: list[PlayerDetectionResult] = []

        for frame_data, result in zip(frames, raw_results):
            detected_objects: list[DetectedObject] = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    # Safely convert PyTorch tensors or numpy arrays
                    raw_xyxy = box.xyxy[0]
                    xyxy = raw_xyxy.cpu().numpy() if hasattr(raw_xyxy, "cpu") else raw_xyxy

                    raw_conf = box.conf[0]
                    conf = float(raw_conf.cpu().numpy() if hasattr(raw_conf, "cpu") else raw_conf)

                    raw_cls = box.cls[0]
                    cls_id = int(raw_cls.cpu().numpy() if hasattr(raw_cls, "cpu") else raw_cls)

                    if cls_id == COCO_PERSON_CLASS_ID and conf >= self.confidence_threshold:
                        bbox = BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=round(conf, 4),
                        )
                        detected_objects.append(
                            DetectedObject(
                                object_type=TrackedObjectType.PLAYER,
                                class_id=cls_id,
                                confidence=round(conf, 4),
                                bbox=bbox,
                            )
                        )

            detection_results.append(
                PlayerDetectionResult(
                    frame_number=frame_data.frame_number,
                    timestamp_seconds=frame_data.timestamp_seconds,
                    detections=detected_objects,
                    inference_time_ms=round(per_frame_time_ms, 2),
                )
            )

        return detection_results
