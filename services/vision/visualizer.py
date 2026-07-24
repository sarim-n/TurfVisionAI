"""
Purpose: Visualization helper for rendering player detection bounding boxes on OpenCV frames.
Dependencies: cv2, numpy, services.vision.models
Inputs: Raw image frame, PlayerDetectionResult
Outputs: Annotated BGR image frame
"""

import cv2
import numpy as np
from services.vision.models import PlayerDetectionResult


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
