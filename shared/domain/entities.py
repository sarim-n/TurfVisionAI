"""
Purpose: Core domain entities and value objects for the TurfVision AI system.
Dependencies: pydantic
Inputs: Primitive types (floats, ints, strings)
Outputs: Immutable domain DTOs (BoundingBox, Point2D, MatchState)
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TrackedObjectType(str, Enum):
    PLAYER = "player"
    BALL = "ball"
    REFEREE = "referee"
    GOAL = "goal"


class TeamSide(str, Enum):
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class Point2D(BaseModel):
    """Immutable 2D coordinate on pixel canvas or 2D pitch ground plane."""
    x: float
    y: float

    @property
    def x_meters(self) -> float:
        return self.x

    @property
    def y_meters(self) -> float:
        return self.y


class BoundingBox(BaseModel):
    """Bounding box representation in standard xyxy coordinate format."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D(x=self.x1 + self.width / 2.0, y=self.y1 + self.height / 2.0)

    @property
    def ground_position(self) -> Point2D:
        """Ground contact point (bottom-center of bounding box) for player feet."""
        return Point2D(x=self.x1 + self.width / 2.0, y=self.y2)

    @property
    def bottom_center(self) -> Point2D:
        return self.ground_position

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0


class MatchState(BaseModel):
    """Immutable domain snapshot representing live match scoreboard and period status."""
    match_id: str = "match_01"
    home_score: int = 0
    away_score: int = 0
    elapsed_seconds: float = 0.0
    possession_team: TeamSide = TeamSide.UNKNOWN
    current_period: str = "1st_half"
    is_active: bool = False
