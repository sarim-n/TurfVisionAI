"""
Purpose: Data DTO models for match statistics, player telemetry, and analytics reports.
Dependencies: pydantic, shared.domain.entities
Inputs: Aggregated tracking data
Outputs: Validated PlayerStats, TeamStats, and MatchAnalyticsReport models
"""

from pydantic import BaseModel, Field
from shared.domain.entities import TeamSide


class PlayerStats(BaseModel):
    """Detailed analytical metrics for an individual player."""
    track_id: int = Field(..., description="Unique persistent track ID")
    team: TeamSide = Field(default=TeamSide.UNKNOWN, description="Player team side")
    distance_run_meters: float = Field(default=0.0, ge=0.0, description="Total distance covered in meters")
    avg_speed_kmh: float = Field(default=0.0, ge=0.0, description="Average movement speed in km/h")
    max_speed_kmh: float = Field(default=0.0, ge=0.0, description="Peak sprint speed in km/h")
    passes_attempted: int = Field(default=0, ge=0)
    passes_completed: int = Field(default=0, ge=0)
    shots: int = Field(default=0, ge=0)
    goals: int = Field(default=0, ge=0)


class TeamStats(BaseModel):
    """Aggregated analytical metrics for a team."""
    team: TeamSide
    possession_percentage: float = Field(default=50.0, ge=0.0, le=100.0)
    total_distance_run_meters: float = Field(default=0.0, ge=0.0)
    total_passes: int = Field(default=0, ge=0)
    pass_accuracy_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    shots_total: int = Field(default=0, ge=0)
    shots_on_target: int = Field(default=0, ge=0)


class MatchAnalyticsReport(BaseModel):
    """Comprehensive post-match or live analytical report."""
    match_id: str
    home_stats: TeamStats
    away_stats: TeamStats
    player_stats: list[PlayerStats] = Field(default_factory=list)
