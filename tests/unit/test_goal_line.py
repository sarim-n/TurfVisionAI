"""
Unit tests for Goal Line Spatial Detection Engine.
"""

import numpy as np
import pytest
from services.vision.goal_line import GoalLineDetector
from services.vision.models import GoalSide, TrackedBall
from services.vision.visualizer import draw_goal_lines
from shared.domain.entities import Point2D


def test_goal_line_perpendicular_distance():
    detector = GoalLineDetector.create_default_detector(image_width=640, image_height=480)

    # Home goal line is vertical at x = 50.0, from y = 180 to y = 300
    # Ball 1: At (20, 240) -> In front of line (x=20) -> Signed distance = 30.0 px
    ball_in_front = TrackedBall(center=Point2D(x=20.0, y=240.0))
    res1 = detector.check_goal_line_crossing(ball_in_front, GoalSide.HOME_GOAL)
    assert res1.is_ball_in_goal_mouth is True
    assert abs(res1.perpendicular_distance_px - 30.0) < 0.1
    assert res1.is_ball_past_goal_line is False

    # Ball 2: At (80, 240) -> Past line inside net (x=80) -> Signed distance = 30.0 px > ball_radius_px (10)
    ball_past = TrackedBall(center=Point2D(x=80.0, y=240.0))
    res2 = detector.check_goal_line_crossing(ball_past, GoalSide.HOME_GOAL)
    assert res2.is_ball_in_goal_mouth is True
    assert res2.is_ball_past_goal_line is True

    # Ball 3: At (80, 10) -> Outside goal mouth (y=10) -> is_ball_in_goal_mouth = False
    ball_outside = TrackedBall(center=Point2D(x=80.0, y=10.0))
    res3 = detector.check_goal_line_crossing(ball_outside, GoalSide.HOME_GOAL)
    assert res3.is_ball_in_goal_mouth is False
    assert res3.is_ball_past_goal_line is False


def test_draw_goal_lines():
    detector = GoalLineDetector.create_default_detector(640, 480)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = draw_goal_lines(image, detector)
    assert annotated.shape == (480, 640, 3)
