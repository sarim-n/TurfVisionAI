"""
Purpose: Goal Line Spatial Detection Engine.
Dependencies: math, numpy, services.vision.models, services.vision.homography, shared.domain.entities, shared.logging
Inputs: TrackedBall, GoalSide, GoalPostGeometry, and PitchHomography
Outputs: GoalLineCheckResult assessing goal mouth alignment and goal line plane crossing
"""

import math
from typing import Optional
from services.vision.homography import PitchHomography
from services.vision.models import GoalLineCheckResult, GoalPostGeometry, GoalSide, TrackedBall
from shared.domain.entities import Point2D
from shared.logging import setup_logger

logger = setup_logger("goal_line", service_name="vision")


class GoalLineDetector:
    """Evaluates 2D/3D spatial coordinates of tracked football relative to goal post plane lines."""

    def __init__(self, home_goal: GoalPostGeometry, away_goal: GoalPostGeometry):
        self.goals: dict[GoalSide, GoalPostGeometry] = {
            GoalSide.HOME_GOAL: home_goal,
            GoalSide.AWAY_GOAL: away_goal,
        }

    def check_goal_line_crossing(
        self,
        tracked_ball: TrackedBall,
        goal_side: GoalSide,
        homography: Optional[PitchHomography] = None,
        ball_radius_px: float = 10.0,
    ) -> GoalLineCheckResult:
        """Evaluates signed distance and goal line crossing state for a tracked ball."""
        goal_geom = self.goals[goal_side]
        p1 = goal_geom.left_post
        p2 = goal_geom.right_post

        bx = tracked_ball.center.x
        by = tracked_ball.center.y

        # Line equation: a*x + b*y + c = 0
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = p2.x * p1.y - p1.x * p2.y

        denom = math.sqrt(a**2 + b**2)
        if denom == 0:
            perpendicular_dist = 0.0
            signed_dist = 0.0
        else:
            signed_dist = (a * bx + b * by + c) / denom
            perpendicular_dist = abs(signed_dist)

        # Check if ball center is within left/right post span (projected onto goal line vector)
        v_line = (p2.x - p1.x, p2.y - p1.y)
        v_ball = (bx - p1.x, by - p1.y)
        line_len_sq = v_line[0] ** 2 + v_line[1] ** 2

        if line_len_sq > 0:
            projection_t = (v_ball[0] * v_line[0] + v_ball[1] * v_line[1]) / line_len_sq
            is_in_goal_mouth = 0.0 <= projection_t <= 1.0
        else:
            is_in_goal_mouth = False

        # Orientation convention: positive signed distance means ball has crossed plane into net
        # 100% of ball must cross -> distance past line > ball_radius_px
        is_past_line = is_in_goal_mouth and (signed_dist > ball_radius_px)

        # Compute metric distance if homography is available
        if homography is not None and tracked_ball.pitch_position is not None:
            # Metric signed distance (meters)
            if goal_side == GoalSide.HOME_GOAL:
                # Home goal at x = 0.0m -> past line if x < 0.0 or x > length
                dist_meters = -tracked_ball.pitch_position.x_meters
            else:
                # Away goal at x = 105.0m
                dist_meters = tracked_ball.pitch_position.x_meters - homography.pitch_dimensions.length_meters
        else:
            # Fallback px to meters estimation (assuming 1m = ~30px)
            dist_meters = round(signed_dist / 30.0, 3)

        return GoalLineCheckResult(
            goal_side=goal_side,
            is_ball_in_goal_mouth=is_in_goal_mouth,
            is_ball_past_goal_line=is_past_line,
            signed_distance_meters=dist_meters,
            perpendicular_distance_px=round(perpendicular_dist, 2),
        )

    @classmethod
    def create_default_detector(cls, image_width: int, image_height: int) -> "GoalLineDetector":
        """Creates default Home and Away goal post geometries for testing/calibration."""
        home_left = Point2D(x=50.0, y=float(image_height // 2 - 60))
        home_right = Point2D(x=50.0, y=float(image_height // 2 + 60))
        home_goal = GoalPostGeometry(
            goal_side=GoalSide.HOME_GOAL,
            left_post=home_left,
            right_post=home_right,
            crossbar_height_px=80.0,
        )

        away_left = Point2D(x=float(image_width - 50), y=float(image_height // 2 - 60))
        away_right = Point2D(x=float(image_width - 50), y=float(image_height // 2 + 60))
        away_goal = GoalPostGeometry(
            goal_side=GoalSide.AWAY_GOAL,
            left_post=away_left,
            right_post=away_right,
            crossbar_height_px=80.0,
        )

        return cls(home_goal, away_goal)
