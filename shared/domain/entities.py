"""
Purpose: Defines domain entities and value objects for the football vision pipeline.
Dependencies: pydantic, enum
Inputs: Object coordinates, track identifiers, game status
Outputs: Validated domain entities
"""

from enum import Enum
from pydantic import BaseModel, Field


class TrackedObjectType(str, Enum):
    PLAYER = "player"
    BALL = "ball"
    REFEREE = "referee"
    GOAL_POST = "goal_post"


class TeamSide(str, Enum):
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class Point2D(BaseModel):
    """2D Spatial Coordinate (pixel space or bird's eye pitch space)."""
    x: float
    y: float


class BoundingBox(BaseModel):
    """Axis-Aligned Bounding Box (xyxy format)."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def center(self) -> Point2D:
        """Computes the center point of the bounding box."""
        return Point2D(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point2D:
        """Computes the bottom center point (ideal for player ground position)."""
        return Point2D(x=(self.x1 + self.x2) / 2.0, y=self.y2)


class MatchState(BaseModel):
    """Represents live match state telemetry."""
    match_id: str
    home_score: int = 0
    away_score: int = 0
    elapsed_seconds: float = 0.0
    possession_team: TeamSide = TeamSide.UNKNOWN
    is_active: bool = True
