"""
Unit tests for Goal Detection Event Engine.
"""

import numpy as np
import pytest
from services.event_engine.goal_engine import GoalEngine
from services.ingestion.frame import FrameData
from services.ingestion.metadata import VideoMetadata
from services.vision.models import GoalLineCheckResult, GoalSide, TrackedBall
from shared.domain.entities import Point2D, TeamSide
from shared.schemas.events import EventType


@pytest.fixture
def sample_frame_data():
    metadata = VideoMetadata(
        source_path="test.mp4",
        width=640,
        height=480,
        fps=30.0,
        total_frames=100,
        duration_seconds=3.33,
        aspect_ratio=1.333,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return FrameData(
        frame_number=100,
        timestamp_seconds=3.33,
        image=image,
        metadata=metadata,
    )


def test_goal_engine_event_emission_and_cooldown(sample_frame_data):
    engine = GoalEngine(match_id="match_test_101", cooldown_seconds=5.0)
    tracked_ball = TrackedBall(
        center=Point2D(x=80.0, y=240.0),
        speed_px_per_sec=150.0,
    )
    goal_check_crossing = GoalLineCheckResult(
        goal_side=GoalSide.HOME_GOAL,
        is_ball_in_goal_mouth=True,
        is_ball_past_goal_line=True,
        signed_distance_meters=1.5,
        perpendicular_distance_px=45.0,
    )

    # Frame 1: Ball crosses line -> Should emit Goal Event for AWAY team
    event1 = engine.process_frame(tracked_ball, goal_check_crossing, None, sample_frame_data)
    assert event1 is not None
    assert event1.event_type == EventType.GOAL
    assert event1.team == TeamSide.AWAY
    assert event1.match_id == "match_test_101"

    # Frame 2: Subsequent frame 0.1s later (within 5s cooldown) -> Should be suppressed (None)
    frame2 = sample_frame_data
    frame2.frame_number = 103
    frame2.timestamp_seconds = 3.43

    event2 = engine.process_frame(tracked_ball, goal_check_crossing, None, frame2)
    assert event2 is None


def test_goal_engine_reset(sample_frame_data):
    engine = GoalEngine(cooldown_seconds=5.0)
    tracked_ball = TrackedBall(center=Point2D(x=80.0, y=240.0))
    goal_check = GoalLineCheckResult(
        goal_side=GoalSide.HOME_GOAL,
        is_ball_in_goal_mouth=True,
        is_ball_past_goal_line=True,
        signed_distance_meters=1.5,
        perpendicular_distance_px=45.0,
    )

    event1 = engine.process_frame(tracked_ball, goal_check, None, sample_frame_data)
    assert event1 is not None

    # Reset engine
    engine.reset()

    # Frame 2: Immediately process frame again after reset -> Should emit new Goal Event
    event2 = engine.process_frame(tracked_ball, goal_check, None, sample_frame_data)
    assert event2 is not None
