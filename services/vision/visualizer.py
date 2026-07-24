"""
Purpose: Visualization helper for rendering player detections, ball detections, player tracks, ball tracking vectors, 2D tactical maps, goal line wireframes, and TV scoreboards on OpenCV frames.
Dependencies: cv2, numpy, services.vision.models, services.vision.homography, services.vision.goal_line, services.event_engine.scoreboard
Inputs: Raw image frame, tracking results, GoalLineDetector, ScoreboardEngine
Outputs: Annotated BGR image frame
"""

from typing import Optional
import cv2
import numpy as np
from services.event_engine.scoreboard import ScoreboardEngine
from services.vision.goal_line import GoalLineDetector
from services.vision.homography import PitchHomography
from services.vision.models import (
    BallDetectionResult,
    BallTrackingFrameResult,
    GoalSide,
    PlayerDetectionResult,
    PlayerTrackingFrameResult,
    TrackedBall,
    TrackedPlayer,
)


def draw_player_detections(
    image: np.ndarray,
    result: PlayerDetectionResult,
    box_color: tuple[int, int, int] = (0, 255, 0),
    draw_foot_point: bool = True,
) -> np.ndarray:
    """Draws player detection bounding boxes, confidence text, and ground foot position dots."""
    annotated = image.copy()

    for detection in result.detections:
        bbox = detection.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        # Draw bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Draw confidence label badge
        label_text = f"Player {detection.confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - h - 6), (x1 + w + 6, y1), box_color, -1)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw ground foot position dot
        if draw_foot_point:
            foot_x = int((bbox.x1 + bbox.x2) / 2.0)
            foot_y = int(bbox.y2)
            cv2.circle(annotated, (foot_x, foot_y), 4, (0, 0, 255), -1)

    return annotated


def draw_ball_detections(
    image: np.ndarray,
    result: BallDetectionResult,
    ball_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draws football detection circle highlight and target marker on OpenCV image arrays."""
    annotated = image.copy()

    if result.has_ball and result.ball_object is not None:
        bbox = result.ball_object.bbox
        center_x = int(bbox.center.x)
        center_y = int(bbox.center.y)
        radius = max(6, int(max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1) / 2.0))

        # Draw highlighted circle around football
        cv2.circle(annotated, (center_x, center_y), radius + 3, ball_color, 2)
        cv2.circle(annotated, (center_x, center_y), 2, (0, 0, 255), -1)

        # Label text
        label_text = f"Ball {result.ball_object.confidence:.2f}"
        cv2.putText(
            annotated,
            label_text,
            (center_x + radius + 5, center_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            ball_color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_player_tracks(
    image: np.ndarray,
    result: PlayerTrackingFrameResult,
    draw_tail: bool = True,
) -> np.ndarray:
    """Draws tracked players with persistent ID badges (#X) and ground motion trajectory tail lines."""
    annotated = image.copy()

    for player in result.tracked_players:
        bbox = player.bbox
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)

        # Generate consistent color based on track_id
        np.random.seed(player.track_id)
        color = tuple(map(int, np.random.randint(50, 255, size=3)))

        # Draw player bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw Track ID badge (#1, #2, etc.)
        label_text = f"#{player.track_id}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, y1 - h - 6), (x1 + w + 6, y1), color, -1)
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw motion trajectory tail
        if draw_tail and len(player.trajectory_history) > 1:
            points = [(int(pt.x), int(pt.y)) for pt in player.trajectory_history]
            for i in range(1, len(points)):
                thickness = max(1, int(i / len(points) * 3))
                cv2.line(annotated, points[i - 1], points[i], color, thickness)

    return annotated


def draw_ball_tracks(
    image: np.ndarray,
    result: BallTrackingFrameResult,
    draw_velocity_arrow: bool = True,
    draw_flight_tail: bool = True,
) -> np.ndarray:
    """Draws tracked football, velocity vector arrow, and flight path trajectory tail."""
    annotated = image.copy()

    if result.has_ball and result.tracked_ball is not None:
        tb = result.tracked_ball
        center_x = int(tb.center.x)
        center_y = int(tb.center.y)

        # Color: Bright Yellow for active detection, Magenta for interpolated prediction
        color = (255, 0, 255) if tb.is_interpolated else (0, 255, 255)

        # Draw ball center dot and ring
        cv2.circle(annotated, (center_x, center_y), 6, color, -1)
        cv2.circle(annotated, (center_x, center_y), 9, (0, 0, 0), 2)

        # Draw flight trajectory tail
        if draw_flight_tail and len(tb.trajectory_history) > 1:
            pts = [(int(pt.x), int(pt.y)) for pt in tb.trajectory_history]
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], (0, 255, 255), 2)

        # Draw velocity arrow
        if draw_velocity_arrow and (abs(tb.velocity.x) > 1.0 or abs(tb.velocity.y) > 1.0):
            arrow_end_x = int(center_x + tb.velocity.x * 0.2)
            arrow_end_y = int(center_y + tb.velocity.y * 0.2)
            cv2.arrowedLine(
                annotated,
                (center_x, center_y),
                (arrow_end_x, arrow_end_y),
                (0, 0, 255),
                2,
                tipLength=0.3,
            )

        # Speed text badge
        status = "INTERPOLATED" if tb.is_interpolated else f"{tb.speed_px_per_sec:.0f} px/s"
        cv2.putText(
            annotated,
            f"BALL ({status})",
            (center_x + 12, center_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_birds_eye_view(
    tracked_players: list[TrackedPlayer],
    tracked_ball: Optional[TrackedBall],
    homography: PitchHomography,
    canvas_size: tuple[int, int] = (600, 400),
) -> np.ndarray:
    """Renders a 2D top-down Tactical Pitch map showing player dots, track IDs, and ball position."""
    canvas_w, canvas_h = canvas_size
    margin = 30
    pitch_draw_w = canvas_w - (2 * margin)
    pitch_draw_h = canvas_h - (2 * margin)

    # Green grass canvas background
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = (34, 139, 34)  # Forest Green

    # Pitch boundary box
    cv2.rectangle(
        canvas,
        (margin, margin),
        (margin + pitch_draw_w, margin + pitch_draw_h),
        (255, 255, 255),
        2,
    )
    # Halfway line
    mid_x = margin + pitch_draw_w // 2
    cv2.line(canvas, (mid_x, margin), (mid_x, margin + pitch_draw_h), (255, 255, 255), 2)
    # Center circle
    cv2.circle(canvas, (mid_x, margin + pitch_draw_h // 2), 40, (255, 255, 255), 2)
    cv2.circle(canvas, (mid_x, margin + pitch_draw_h // 2), 4, (255, 255, 255), -1)

    p_dims = homography.pitch_dimensions

    def pitch_to_canvas(px: float, py: float) -> tuple[int, int]:
        cx = int(margin + (px / p_dims.length_meters) * pitch_draw_w)
        cy = int(margin + (py / p_dims.width_meters) * pitch_draw_h)
        return cx, cy

    # Draw tracked players on 2D tactical map
    for player in tracked_players:
        pitch_pt = homography.pixel_to_pitch(player.ground_position)
        cx, cy = pitch_to_canvas(pitch_pt.x_meters, pitch_pt.y_meters)

        np.random.seed(player.track_id)
        color = tuple(map(int, np.random.randint(80, 255, size=3)))

        cv2.circle(canvas, (cx, cy), 7, color, -1)
        cv2.circle(canvas, (cx, cy), 9, (255, 255, 255), 1)
        cv2.putText(
            canvas,
            f"#{player.track_id}",
            (cx + 8, cy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Draw tracked ball on 2D tactical map
    if tracked_ball is not None:
        pitch_pt = homography.pixel_to_pitch(tracked_ball.center)
        bx, by = pitch_to_canvas(pitch_pt.x_meters, pitch_pt.y_meters)
        cv2.circle(canvas, (bx, by), 6, (0, 255, 255), -1)
        cv2.circle(canvas, (bx, by), 8, (0, 0, 0), 2)

    return canvas


def draw_goal_lines(
    image: np.ndarray,
    goal_detector: GoalLineDetector,
    line_color: tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Draws goal post geometry lines and goal line plane segments on OpenCV image frames."""
    annotated = image.copy()

    for goal_side in [GoalSide.HOME_GOAL, GoalSide.AWAY_GOAL]:
        goal_geom = goal_detector.goals[goal_side]
        p1 = (int(goal_geom.left_post.x), int(goal_geom.left_post.y))
        p2 = (int(goal_geom.right_post.x), int(goal_geom.right_post.y))

        # Goal Line Plane
        cv2.line(annotated, p1, p2, line_color, 3)

        # Goal Post Dots
        cv2.circle(annotated, p1, 6, (255, 255, 255), -1)
        cv2.circle(annotated, p2, 6, (255, 255, 255), -1)

        # Label badge
        label = "HOME GOAL" if goal_side == GoalSide.HOME_GOAL else "AWAY GOAL"
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(
            annotated,
            label,
            (mid_x - 30, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_scoreboard_overlay(
    image: np.ndarray,
    scoreboard: ScoreboardEngine,
) -> np.ndarray:
    """Renders a broadcast TV-style scoreboard banner overlay at the top-left of image frames."""
    annotated = image.copy()
    box_x, box_y = 20, 20
    box_w, box_h = 280, 45

    # Dark translucent banner background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # White border frame
    cv2.rectangle(annotated, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1)

    # Team names & score: HOME 2 - 1 AWAY
    score_text = f"{scoreboard.home_team_name} {scoreboard.home_score} - {scoreboard.away_score} {scoreboard.away_team_name}"
    cv2.putText(
        annotated,
        score_text,
        (box_x + 12, box_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Clock & Period status text: 1st Half | 12:45
    clock_text = f"{scoreboard.period.value.upper()} | {scoreboard.format_clock()}"
    cv2.putText(
        annotated,
        clock_text,
        (box_x + 12, box_y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated
