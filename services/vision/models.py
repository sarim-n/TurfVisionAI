"""
Purpose: Data models for object detection, tracking, spatial pitch coordinates, and goal line geometry in the vision service.
Dependencies: pydantic, enum, shared.domain.entities
Inputs: Raw model predictions, tracking updates, pitch calibrations, and goal post geometry
Outputs: Validated domain models for vision pipeline and spatial goal line checks
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from shared.domain.entities import BoundingBox, Point2D, TrackedObjectType


class GoalSide(str, Enum):
    HOME_GOAL = "home_goal"
    AWAY_GOAL = "away_goal"


class PitchDimensions(BaseModel):
    """Real-world dimensions of the football pitch in meters."""
    length_meters: float = Field(default=105.0, description="Standard pitch length in meters")
    width_meters: float = Field(default=68.0, description="Standard pitch width in meters")
    penalty_box_length: float = Field(default=16.5, description="Penalty area length in meters")
    penalty_box_width: float = Field(default=40.32, description="Penalty area width in meters")


class PitchPoint(BaseModel):
    """2D Real-World Coordinate on the football pitch in meters (0,0 is top-left corner)."""
    x_meters: float = Field(..., description="X coordinate along pitch length (0 to length_meters)")
    y_meters: float = Field(..., description="Y coordinate along pitch width (0 to width_meters)")


class GoalPostGeometry(BaseModel):
    """Defines spatial coordinates of goal posts and crossbar for a specific goal side."""
    goal_side: GoalSide
    left_post: Point2D = Field(..., description="Left goal post base pixel coordinate")
    right_post: Point2D = Field(..., description="Right goal post base pixel coordinate")
    crossbar_height_px: float = Field(default=100.0, description="Goal crossbar height in pixels")
    width_meters: float = Field(default=7.32, description="Official goal mouth width in meters")
    height_meters: float = Field(default=2.44, description="Official goal height in meters")


class GoalLineCheckResult(BaseModel):
    """Output of spatial goal line evaluation for a tracked ball."""
    goal_side: GoalSide
    is_ball_in_goal_mouth: bool = Field(..., description="True if ball center is within left/right goal post width")
    is_ball_past_goal_line: bool = Field(..., description="True if 100% of ball has crossed goal line plane into net")
    signed_distance_meters: float = Field(..., description="Signed perpendicular distance in meters (+ is inside net)")
    perpendicular_distance_px: float = Field(..., description="Perpendicular distance in pixels")


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
    pitch_position: Optional[PitchPoint] = Field(default=None, description="2D pitch position in meters")
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
    pitch_position: Optional[PitchPoint] = Field(default=None, description="2D pitch position in meters")
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
