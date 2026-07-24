"""
Purpose: Visualization helper for rendering player detections, ball detections, and player tracking trajectories on OpenCV frames.
Dependencies: cv2, numpy, services.vision.models
Inputs: Raw image frame, PlayerDetectionResult / BallDetectionResult / PlayerTrackingFrameResult
Outputs: Annotated BGR image frame
"""

import cv2
import numpy as np
from services.vision.models import BallDetectionResult, PlayerDetectionResult, PlayerTrackingFrameResult


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
