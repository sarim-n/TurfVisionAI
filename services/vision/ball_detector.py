"""
Purpose: Football Detection Service using YOLO with small-object candidate filtering.
Dependencies: ultralytics, time, services.ingestion.frame, shared.domain.entities
Inputs: FrameData objects
Outputs: BallDetectionResult domain objects containing football bounding box candidates
"""

import time
from typing import Optional
from services.ingestion.frame import FrameData
from services.vision.models import BallDetectionResult, DetectedObject
from shared.config import get_settings
from shared.domain.entities import BoundingBox, TrackedObjectType
from shared.logging import setup_logger

logger = setup_logger("ball_detector", service_name="vision")

# COCO class ID 32 is 'sports ball'
COCO_SPORTS_BALL_CLASS_ID = 32


class BallDetector:
    """YOLO-based Football Detector with spatial aspect-ratio and area sanity candidate filtering."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.20,
        device: Optional[str] = None,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 1.6,
        max_area_ratio: float = 0.05,
    ):
        settings = get_settings()
        self.model_path = model_path or settings.BALL_YOLO_MODEL_PATH
        # Use fallback YOLO_MODEL_PATH if BALL_YOLO_MODEL_PATH weights file doesn't exist yet
        self.confidence_threshold = confidence_threshold
        self.device = device or settings.DEVICE
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_area_ratio = max_area_ratio
        self._model = None

    def _load_model(self) -> None:
        """Lazy loader for Ultralytics YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO  # type: ignore

                logger.info(f"Loading Ball YOLO model from '{self.model_path}' on device '{self.device}'...")
                self._model = YOLO(self.model_path)
                logger.info("Ball YOLO model loaded successfully.")
            except Exception as e:
                # Fallback to standard YOLO model if custom ball model path is not yet present
                settings = get_settings()
                if self.model_path != settings.YOLO_MODEL_PATH:
                    logger.warning(
                        f"Failed to load custom ball model '{self.model_path}'. Falling back to default '{settings.YOLO_MODEL_PATH}'."
                    )
                    self.model_path = settings.YOLO_MODEL_PATH
                    self._model = YOLO(self.model_path)
                else:
                    raise RuntimeError(f"Ball YOLO initialization error: {e}") from e

    def is_valid_ball_bbox(self, bbox: BoundingBox, frame_width: int, frame_height: int) -> bool:
        """Applies spatial aspect ratio and area filters to reject non-ball detections (e.g. white shoes/post)."""
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1

        if w <= 0 or h <= 0:
            return False

        aspect_ratio = w / h
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False

        box_area = w * h
        frame_area = frame_width * frame_height
        if frame_area > 0 and (box_area / frame_area) > self.max_area_ratio:
            return False

        return True

    def detect_ball(self, frame_data: FrameData) -> BallDetectionResult:
        """Detects the football in a single FrameData object."""
        results = self.detect_ball_batch([frame_data])
        return results[0]

    def detect_ball_batch(self, frames: list[FrameData]) -> list[BallDetectionResult]:
        """Performs batch ball detection over a list of FrameData objects."""
        if not frames:
            return []

        self._load_model()
        images = [f.image for f in frames]

        start_time = time.perf_counter()
        # Accept both class 32 (COCO sports ball) and class 0 (custom single-class ball detector)
        raw_results = self._model.predict(
            source=images,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        per_frame_time_ms = total_time_ms / len(frames)

        detection_results: list[BallDetectionResult] = []

        for frame_data, result in zip(frames, raw_results):
            candidates: list[DetectedObject] = []

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    raw_xyxy = box.xyxy[0]
                    xyxy = raw_xyxy.cpu().numpy() if hasattr(raw_xyxy, "cpu") else raw_xyxy

                    raw_conf = box.conf[0]
                    conf = float(raw_conf.cpu().numpy() if hasattr(raw_conf, "cpu") else raw_conf)

                    raw_cls = box.cls[0]
                    cls_id = int(raw_cls.cpu().numpy() if hasattr(raw_cls, "cpu") else raw_cls)

                    # Filter for sports ball class or single-class model
                    if cls_id in (COCO_SPORTS_BALL_CLASS_ID, 0) and conf >= self.confidence_threshold:
                        bbox = BoundingBox(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                            confidence=round(conf, 4),
                        )

                        if self.is_valid_ball_bbox(bbox, frame_data.width, frame_data.height):
                            candidates.append(
                                DetectedObject(
                                    object_type=TrackedObjectType.BALL,
                                    class_id=cls_id,
                                    confidence=round(conf, 4),
                                    bbox=bbox,
                                )
                            )

            # Sort candidates by confidence score descending
            candidates.sort(key=lambda obj: obj.confidence, reverse=True)
            primary_ball = candidates[0] if candidates else None

            detection_results.append(
                BallDetectionResult(
                    frame_number=frame_data.frame_number,
                    timestamp_seconds=frame_data.timestamp_seconds,
                    has_ball=primary_ball is not None,
                    ball_object=primary_ball,
                    candidates=candidates,
                    inference_time_ms=round(per_frame_time_ms, 2),
                )
            )

        return detection_results
