"""
Purpose: Data DTO models for TurfVision AI Pipeline configuration and frame execution results.
Dependencies: pydantic, numpy, shared.domain.entities, shared.schemas.events, services.vision.models
Inputs: Pipeline parameters
Outputs: PipelineConfig and PipelineFrameResult DTO models
"""

from typing import Any, Optional
import numpy as np
from pydantic import BaseModel, Field
from services.event_engine.scoreboard import MatchPeriod
from services.vision.models import TrackedBall, TrackedPlayer
from shared.domain.entities import MatchState
from shared.schemas.events import MatchEvent


class PipelineConfig(BaseModel):
    """Configuration options for TurfVision AI Processing Pipeline."""
    match_id: str = Field(default="match_live_01", description="Match identifier")
    video_source_path: str = Field(default="", description="Path to input video file or RTSP stream")
    output_video_path: str = Field(default="", description="Path to save annotated MP4 video output")
    stride: int = Field(default=1, ge=1, description="Frame stride sampling")
    enable_visualization: bool = Field(default=True, description="Render overlay visuals")
    home_team_name: str = Field(default="HOME FC")
    away_team_name: str = Field(default="AWAY UT")
    yolo_player_model_path: str = Field(default="models/yolov8x.pt")
    yolo_ball_model_path: str = Field(default="models/yolov8x.pt")

    model_config = {"arbitrary_types_allowed": True}


class PipelineFrameResult(BaseModel):
    """DTO containing results for a single processed frame."""
    frame_number: int
    timestamp_seconds: float
    match_state: MatchState
    tracked_players: list[TrackedPlayer] = Field(default_factory=list)
    tracked_ball: Optional[TrackedBall] = None
    new_events: list[MatchEvent] = Field(default_factory=list)
    annotated_image: Optional[Any] = Field(default=None, description="OpenCV image array with overlays")

    model_config = {"arbitrary_types_allowed": True}
