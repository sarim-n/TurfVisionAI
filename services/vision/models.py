"""
Purpose: Data models for object detection and tracking results in the vision service.
Dependencies: pydantic, shared.domain.entities
Inputs: Raw model predictions & tracker updates
Outputs: Validated domain models for player detection, ball detection, player tracking, and ball tracking
"""

from typing import Optional
from pydantic import BaseModel, Field
from shared.domain.entities import BoundingBox, Point2D, TrackedObjectType


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


class TrackedPlayer(BaseModel):
    """Represents a unique tracked player across time."""
    track_id: int = Field(..., description="Unique persistent tracking ID")
    bbox: BoundingBox = Field(..., description="Current frame bounding box")
    ground_position: Point2D = Field(..., description="Current bottom-center ground location")
    trajectory_history: list[Point2D] = Field(
        default_factory=list, description="Historical ground positions (motion tail)"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    active_frames: int = Field(default=1, description="Number of consecutive active frames tracked")


class PlayerTrackingFrameResult(BaseModel):
    """Aggregated player tracking output for a single frame."""
    frame_number: int
    timestamp_seconds: float
    tracked_players: list[TrackedPlayer] = Field(default_factory=list)

    @property
    def active_player_count(self) -> int:
        return len(self.tracked_players)


class TrackedBall(BaseModel):
    """Represents the tracked football with position, velocity, and interpolation status."""
    center: Point2D = Field(..., description="Current ball center (x, y)")
    velocity: Point2D = Field(default_factory=lambda: Point2D(x=0.0, y=0.0), description="Velocity vector (vx, vy)")
    speed_px_per_sec: float = Field(default=0.0, ge=0.0, description="Ball speed in pixels/sec")
    is_interpolated: bool = Field(default=False, description="True if position was predicted due to occlusion")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    trajectory_history: list[Point2D] = Field(
        default_factory=list, description="Historical center positions (flight path)"
    )


class BallTrackingFrameResult(BaseModel):
    """Aggregated ball tracking output for a single frame."""
    frame_number: int
    timestamp_seconds: float
    has_ball: bool = False
    tracked_ball: Optional[TrackedBall] = None
