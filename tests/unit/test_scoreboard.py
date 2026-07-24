"""
Unit tests for Scoreboard Engine & Match State Machine.
"""

import numpy as np
import pytest
from services.event_engine.scoreboard import MatchPeriod, ScoreboardEngine
from services.vision.visualizer import draw_scoreboard_overlay
from shared.domain.entities import TeamSide
from shared.schemas.events import EventType, MatchEvent


def test_scoreboard_engine_clock_and_periods():
    sb = ScoreboardEngine(match_id="match_test", home_team_name="REAL", away_team_name="BARCA")
    assert sb.period == MatchPeriod.NOT_STARTED
    assert sb.format_clock() == "00:00"

    sb.start_match()
    assert sb.period == MatchPeriod.FIRST_HALF

    # Update clock by 90 seconds (1 min 30 sec)
    sb.update_clock(90.0)
    assert sb.format_clock() == "01:30"
    assert sb.elapsed_seconds == 90.0

    sb.pause_match()
    assert sb.period == MatchPeriod.PAUSED
    sb.update_clock(30.0)  # Clock should not advance when paused
    assert sb.format_clock() == "01:30"


def test_scoreboard_score_increment_on_goal():
    sb = ScoreboardEngine(home_team_name="HOME", away_team_name="AWAY")
    sb.start_match()

    # Emit Home Goal
    home_goal_event = MatchEvent(
        event_id="evt_g1",
        match_id="m1",
        event_type=EventType.GOAL,
        video_timestamp_seconds=12.0,
        frame_number=360,
        team=TeamSide.HOME,
    )
    sb.process_event(home_goal_event)
    assert sb.home_score == 1
    assert sb.away_score == 0

    # Emit Away Goal
    away_goal_event = MatchEvent(
        event_id="evt_g2",
        match_id="m1",
        event_type=EventType.GOAL,
        video_timestamp_seconds=45.0,
        frame_number=1350,
        team=TeamSide.AWAY,
    )
    sb.process_event(away_goal_event)
    assert sb.home_score == 1
    assert sb.away_score == 1
    assert len(sb.event_log) == 2

    # Snapshot check
    snapshot = sb.get_state_snapshot()
    assert snapshot.home_score == 1
    assert snapshot.away_score == 1


def test_draw_scoreboard_overlay():
    sb = ScoreboardEngine(home_team_name="FCB", away_team_name="RMA")
    sb.start_match()
    sb.update_clock(75.0)

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_scoreboard_overlay(image, sb)
    assert annotated.shape == (480, 640, 3)
