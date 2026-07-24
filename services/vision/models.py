"""
Purpose: Data models for object detection results in the vision service.
Dependencies: pydantic, shared.domain.entities
Inputs: Raw model predictions
Outputs: Validated DetectedObject, PlayerDetectionResult, and BallDetectionResult domain models
"""

from typing import Optional
from pydantic import BaseModel, Field
from shared.domain.entities import BoundingBox, TrackedObjectType


class DetectedObject(BaseModel):
    """Represents a single detected entity in a video frame."""
    object_type: TrackedObjectType = Field(default=TrackedObjectType.PLAYER)
    class_id: int = Field(..., description="YOLO class ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box in pixel space (x1, y1, x2, y2)")


class PlayerDetectionResult(BaseModel):
    """Aggregated player detection result for a single video frame."""
    frame_number: int
    timestamp_seconds: float
    detections: list[DetectedObject] = Field(default_factory=list)
    inference_time_ms: float = Field(default=0.0, description="Inference execution duration in ms")

    @property
    def player_count(self) -> int:
        """Returns total number of detected players/persons."""
        return len(self.detections)


class BallDetectionResult(BaseModel):
    """Aggregated football detection result for a single video frame."""
    frame_number: int
    timestamp_seconds: float
    has_ball: bool = False
    ball_object: Optional[DetectedObject] = None
    candidates: list[DetectedObject] = Field(default_factory=list)
    inference_time_ms: float = Field(default=0.0, description="Inference execution duration in ms")
