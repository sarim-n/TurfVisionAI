"""
Purpose: Visualization helper for rendering player and ball detection annotations on OpenCV frames.
Dependencies: cv2, numpy, services.vision.models
Inputs: Raw image frame, PlayerDetectionResult / BallDetectionResult
Outputs: Annotated BGR image frame
"""

import cv2
import numpy as np
from services.vision.models import BallDetectionResult, PlayerDetectionResult


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
