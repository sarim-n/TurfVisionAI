"""
Purpose: Data Transfer Objects (DTOs) for Match Events fired by the Event Engine.
Dependencies: pydantic, enum, datetime
Inputs: Event parameters and spatial data
Outputs: Validated JSON event schema
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from shared.domain.entities import Point2D, TeamSide


class EventType(str, Enum):
    GOAL = "goal"
    SHOT = "shot"
    PASS = "pass"
    POSSESSION_CHANGE = "possession_change"
    BALL_OUT = "ball_out"
    HALFTIME = "halftime"
    FULLTIME = "fulltime"


class MatchEvent(BaseModel):
    """Schema representing a validated high-level match event."""
    event_id: str = Field(..., description="Unique event identifier (UUID)")
    match_id: str = Field(..., description="Target match ID")
    event_type: EventType = Field(..., description="Category of event")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    video_timestamp_seconds: float = Field(..., description="Timestamp in match video seconds")
    frame_number: int = Field(..., description="Frame index when event occurred")
    team: TeamSide = Field(default=TeamSide.UNKNOWN, description="Team associated with event")
    player_id: int | None = Field(default=None, description="Tracking ID of primary player")
    location: Point2D | None = Field(default=None, description="Pitch/Frame location of event")
    details: dict[str, Any] = Field(default_factory=dict, description="Metadata specific to event type")
