"""
Purpose: OpenCV visualization module for rendering player tracks, ball velocity vectors, bird's eye tactical pitch view, goal lines, and TV broadcast scoreboard overlays.
Dependencies: cv2, numpy, services.vision.models, shared.domain.entities, shared.schemas.events
Inputs: Raw OpenCV BGR image arrays, tracking DTOs, and match state objects
Outputs: Annotated BGR image arrays with custom overlays
"""

import math
from typing import Optional, Union
import cv2
import numpy as np
from services.event_engine.scoreboard import ScoreboardEngine
from services.vision.goal_line import GoalLineDetector
from services.vision.homography import PitchHomography
from services.vision.models import (
    BallDetectionResult,
    BallTrackingFrameResult,
    GoalLineCheckResult,
    GoalPostGeometry,
    PlayerDetectionResult,
    PlayerTrackingFrameResult,
    TrackedBall,
    TrackedPlayer,
)
from shared.domain.entities import MatchState, Point2D


def draw_player_detections(image: np.ndarray, result: PlayerDetectionResult) -> np.ndarray:
    """Draws raw player detections on image with confidence badge."""
    annotated = image.copy()
    for det in result.detections:
        box = det.bbox
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ground feet contact point
        gp = det.ground_position
        if gp:
            cv2.circle(annotated, (int(gp.x), int(gp.y)), 4, (0, 0, 255), -1)

        # Confidence label
        label = f"Player {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return annotated


def draw_ball_detections(image: np.ndarray, result: BallDetectionResult) -> np.ndarray:
    """Draws detected football with high-visibility yellow highlight circle."""
    annotated = image.copy()
    ball_obj = result.detected_ball if hasattr(result, "detected_ball") else getattr(result, "ball_object", None)
    has_b = result.has_ball if hasattr(result, "has_ball") else (ball_obj is not None)

    if has_b and ball_obj:
        box = ball_obj.bbox
        center_x, center_y = int(box.center.x), int(box.center.y)
        radius = int(max(box.width, box.height) / 2.0) + 4

        # Bright yellow highlight circle
        cv2.circle(annotated, (center_x, center_y), radius, (0, 255, 255), 2)
        cv2.circle(annotated, (center_x, center_y), 3, (0, 0, 255), -1)

        label = f"Ball {ball_obj.confidence:.2f}"
        cv2.putText(annotated, label, (center_x - 20, max(15, center_y - radius - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return annotated


def draw_player_tracks(
    image: np.ndarray,
    result: Union[PlayerTrackingFrameResult, list[TrackedPlayer]],
    draw_tail: bool = True,
) -> np.ndarray:
    """Draws tracked players with persistent ID badges (#X) and ground motion trajectory tail lines."""
    annotated = image.copy()
    players = result.tracked_players if isinstance(result, PlayerTrackingFrameResult) else result

    for player in players:
        box = player.bbox
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)

        # Persistent track ID badge
        badge_text = f"#{player.track_id}"
        cv2.rectangle(annotated, (x1, y1 - 18), (x1 + 40, y1), (255, 128, 0), -1)
        cv2.putText(annotated, badge_text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        # Ground trajectory tail
        if draw_tail and len(player.trajectory_history) > 1:
            pts = [(int(p.x), int(p.y)) for p in player.trajectory_history[-15:]]
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], (0, 255, 255), 2)

    return annotated


def draw_ball_tracks(
    image: np.ndarray,
    result: Union[BallTrackingFrameResult, TrackedBall, None],
) -> np.ndarray:
    """Draws tracked football with Kalman velocity direction arrow and flight trajectory trail."""
    annotated = image.copy()
    if result is None:
        return annotated

    tracked_ball = result.tracked_ball if isinstance(result, BallTrackingFrameResult) else result
    if tracked_ball is None or not hasattr(tracked_ball, "center"):
        return annotated

    cx, cy = int(tracked_ball.center.x), int(tracked_ball.center.y)
    color = (255, 0, 255) if tracked_ball.is_interpolated else (0, 255, 255)

    # Ball marker
    cv2.circle(annotated, (cx, cy), 6, color, -1)
    cv2.circle(annotated, (cx, cy), 8, (255, 255, 255), 1)

    # Velocity direction arrow vector
    vx, vy = tracked_ball.velocity_x, tracked_ball.velocity_y
    if abs(vx) > 0.5 or abs(vy) > 0.5:
        arrow_end = (int(cx + vx * 0.5), int(cy + vy * 0.5))
        cv2.arrowedLine(annotated, (cx, cy), arrow_end, (0, 0, 255), 2, tipLength=0.3)

    # Flight trajectory tail
    if len(tracked_ball.trajectory_history) > 1:
        pts = [(int(p.x), int(p.y)) for p in tracked_ball.trajectory_history[-20:]]
        for i in range(1, len(pts)):
            cv2.line(annotated, pts[i - 1], pts[i], (0, 255, 255), 1)

    return annotated


def draw_goal_lines(
    image: np.ndarray,
    goal: Union[GoalPostGeometry, GoalLineDetector],
    check_result: Optional[GoalLineCheckResult] = None,
) -> np.ndarray:
    """Draws 2D goal post geometry and goal mouth plane lines."""
    annotated = image.copy()
    if isinstance(goal, GoalLineDetector):
        annotated = draw_goal_lines(annotated, goal.left_goal, check_result)
        return draw_goal_lines(annotated, goal.right_goal, check_result)

    p1 = (int(goal.left_post.x), int(goal.left_post.y))
    p2 = (int(goal.right_post.x), int(goal.right_post.y))

    # Goal line color (Red if crossed, Green if clear)
    line_color = (0, 0, 255) if (check_result and check_result.is_ball_past_goal_line) else (0, 255, 0)
    cv2.line(annotated, p1, p2, line_color, 3)

    # Post markers
    cv2.circle(annotated, p1, 6, (255, 255, 255), -1)
    cv2.circle(annotated, p2, 6, (255, 255, 255), -1)

    # Side label badge
    cv2.putText(
        annotated,
        goal.side.value.upper(),
        (min(p1[0], p2[0]) - 30, min(p1[1], p2[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        line_color,
        2,
    )

    return annotated


def draw_scoreboard_overlay(
    image: np.ndarray,
    match_state: Union[MatchState, ScoreboardEngine],
    clock_formatted: str = "00:00",
) -> np.ndarray:
    """Renders a TV broadcast-style scoreboard overlay in top-left corner."""
    annotated = image.copy()
    if isinstance(match_state, ScoreboardEngine):
        home_name = match_state.home_team_name
        away_name = match_state.away_team_name
        home_sc = match_state.home_score
        away_sc = match_state.away_score
        period_val = getattr(match_state.current_period, "value", str(match_state.current_period)).upper()
        clock_str = match_state.format_clock()
    else:
        home_name = getattr(match_state, "home_team_name", "HOME")
        away_name = getattr(match_state, "away_team_name", "AWAY")
        home_sc = getattr(match_state, "home_score", 0)
        away_sc = getattr(match_state, "away_score", 0)
        period_val = str(getattr(match_state, "current_period", "1ST_HALF")).upper()
        clock_str = clock_formatted

    cv2.rectangle(annotated, (20, 20), (320, 70), (15, 23, 42), -1)
    cv2.rectangle(annotated, (20, 20), (320, 70), (255, 255, 255), 1)

    # Team names & score
    score_text = f"{home_name} {home_sc} - {away_sc} {away_name}"
    cv2.putText(annotated, score_text, (30, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Clock & Period
    clock_text = f"{period_val} | {clock_str}"
    cv2.putText(annotated, clock_text, (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (34, 197, 94), 1)

    return annotated


def draw_birds_eye_view(
    players: list[TrackedPlayer],
    ball: Optional[TrackedBall] = None,
    homography: Optional[PitchHomography] = None,
    canvas_size: tuple[int, int] = (600, 400),
) -> np.ndarray:
    """Renders a 2D tactical bird's eye view pitch diagram."""
    w, h = canvas_size
    pitch = np.full((h, w, 3), (34, 100, 34), dtype=np.uint8)  # Forest green pitch

    # Boundary lines & center circle
    cv2.rectangle(pitch, (20, 20), (w - 20, h - 20), (255, 255, 255), 2)
    cv2.line(pitch, (w // 2, 20), (w // 2, h - 20), (255, 255, 255), 2)
    cv2.circle(pitch, (w // 2, h // 2), 40, (255, 255, 255), 2)

    # Draw players
    for p in players:
        if homography:
            pitch_pos = homography.transform_pixel_to_pitch(p.ground_position)
            px = int(min(w - 25, max(25, pitch_pos.x * (w / 105.0))))
            py = int(min(h - 25, max(25, pitch_pos.y * (h / 68.0))))
        else:
            px = int(min(w - 25, max(25, p.ground_position.x * (w / 640.0))))
            py = int(min(h - 25, max(25, p.ground_position.y * (h / 480.0))))
        cv2.circle(pitch, (px, py), 5, (255, 255, 255), -1)

    # Draw ball
    if ball:
        if homography:
            ball_pitch = homography.transform_pixel_to_pitch(ball.center)
            bx = int(min(w - 25, max(25, ball_pitch.x * (w / 105.0))))
            by = int(min(h - 25, max(25, ball_pitch.y * (h / 68.0))))
        else:
            bx = int(min(w - 25, max(25, ball.center.x * (w / 640.0))))
            by = int(min(h - 25, max(25, ball.center.y * (h / 480.0))))
        cv2.circle(pitch, (bx, by), 4, (0, 255, 255), -1)

    return pitch
