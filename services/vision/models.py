"""
Purpose: Domain Data Transfer Objects (DTOs) for computer vision detections and tracking results.
Dependencies: pydantic, shared.domain.entities
Inputs: BoundingBox, Point2D, TrackedObjectType
Outputs: DetectedObject, PlayerDetectionResult, BallDetectionResult, TrackedPlayer, PlayerTrackingFrameResult, etc.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from shared.domain.entities import BoundingBox, Point2D, TrackedObjectType


class GoalSide(str, Enum):
    HOME_GOAL = "home_goal"
    AWAY_GOAL = "away_goal"
    LEFT = "left"
    RIGHT = "right"


class DetectedObject(BaseModel):
    """Domain model representing an object detected by YOLO models."""
    object_type: TrackedObjectType
    class_id: int = Field(default=0, description="COCO or custom YOLO class ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box on pixel image canvas")
    ground_position: Optional[Point2D] = Field(default=None, description="Ground contact position")


class PlayerDetectionResult(BaseModel):
    """Result payload for player detections on a frame."""
    frame_number: int
    timestamp_seconds: float = Field(default=0.0)
    detections: list[DetectedObject] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)

    @property
    def player_count(self) -> int:
        return len(self.detections)


class BallDetectionResult(BaseModel):
    """Result payload for ball detections on a frame."""
    frame_number: int
    timestamp_seconds: float = Field(default=0.0)
    has_ball: bool = Field(default=False)
    ball_object: Optional[DetectedObject] = Field(default=None)
    candidates: list[BoundingBox] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)

    @property
    def detected_ball(self) -> Optional[DetectedObject]:
        return self.ball_object


class TrackedPlayer(BaseModel):
    """Persistent multi-object track for an individual player."""
    track_id: int
    bbox: BoundingBox
    ground_position: Point2D
    pitch_position: Optional[Point2D] = Field(default=None)
    trajectory_history: list[Point2D] = Field(default_factory=list)
    confidence: float = Field(default=1.0)
    active_frames: int = Field(default=1)


class PlayerTrackingFrameResult(BaseModel):
    """Frame result payload from PlayerTracker."""
    frame_number: int = Field(default=0)
    timestamp_seconds: float = Field(default=0.0)
    tracked_players: list[TrackedPlayer] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)

    @property
    def active_player_count(self) -> int:
        return len(self.tracked_players)


class TrackedBall(BaseModel):
    """Kalman filter track state for the football."""
    center: Point2D
    velocity_x: float = Field(default=0.0)
    velocity_y: float = Field(default=0.0)
    speed_px_per_sec: float = Field(default=0.0)
    is_interpolated: bool = Field(default=False)
    trajectory_history: list[Point2D] = Field(default_factory=list)
    pitch_position: Optional[Point2D] = Field(default=None)

    @property
    def velocity(self) -> Point2D:
        return Point2D(x=self.velocity_x, y=self.velocity_y)


class BallTrackingFrameResult(BaseModel):
    """Frame result payload from BallTracker."""
    frame_number: int = Field(default=0)
    timestamp_seconds: float = Field(default=0.0)
    tracked_ball: Optional[TrackedBall] = Field(default=None)
    processing_time_ms: float = Field(default=0.0)

    @property
    def has_ball(self) -> bool:
        return self.tracked_ball is not None


class PitchDimensions(BaseModel):
    length_meters: float = Field(default=105.0)
    width_meters: float = Field(default=68.0)


class PitchPoint(BaseModel):
    x_meters: float
    y_meters: float


class GoalPostGeometry(BaseModel):
    goal_side: GoalSide
    left_post: Point2D
    right_post: Point2D
    crossbar_height_px: float = Field(default=80.0)

    @property
    def side(self) -> GoalSide:
        return self.goal_side

    @property
    def post1(self) -> Point2D:
        return self.left_post

    @property
    def post2(self) -> Point2D:
        return self.right_post

    @property
    def line_a(self) -> float:
        return self.right_post.y - self.left_post.y

    @property
    def line_b(self) -> float:
        return self.left_post.x - self.right_post.x

    @property
    def line_c(self) -> float:
        return self.right_post.x * self.left_post.y - self.left_post.x * self.right_post.y


class GoalLineCheckResult(BaseModel):
    goal_side: GoalSide
    is_ball_in_goal_mouth: bool = Field(default=False)
    is_ball_past_goal_line: bool = Field(default=False)
    signed_distance_meters: float = Field(default=0.0)
    perpendicular_distance_px: float = Field(default=0.0)
    ball_position: Optional[Point2D] = None

    @property
    def side(self) -> GoalSide:
        return self.goal_side

    @property
    def is_past_goal_line(self) -> bool:
        return self.is_ball_past_goal_line

    @property
    def signed_distance(self) -> float:
        return self.signed_distance_meters
