"""
Purpose: Multi-object Kalman filter ball tracker with linear motion state prediction and trajectory smoothing.
Dependencies: numpy, services.vision.models, shared.domain.entities, shared.logging
Inputs: Ball BoundingBox or BallDetectionResult objects, FrameData
Outputs: BallTrackingFrameResult with TrackedBall state (center, velocity, trajectory history)
"""

import math
from typing import Optional, Union
import numpy as np
from services.ingestion.frame import FrameData
from services.vision.models import BallDetectionResult, BallTrackingFrameResult, TrackedBall
from shared.domain.entities import BoundingBox, Point2D
from shared.logging import setup_logger

logger = setup_logger("ball_tracker", service_name="vision")


class BallTracker:
    """Kalman filter-backed single ball tracking pipeline with trajectory history and missing frame prediction."""

    def __init__(self, max_missing_frames: int = 5):
        self.max_missing_frames = max_missing_frames
        self.missing_frames_count = 0
        self.trajectory_history: list[Point2D] = []

        # 4D Kalman Filter State [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)

        # Transition matrix F
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = 1.0  # x = x + vx * dt
        self.F[1, 3] = 1.0  # y = y + vy * dt

        # Measurement matrix H [x, y]
        self.H = np.zeros((2, 4), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # State Covariance P
        self.P = np.eye(4, dtype=np.float32) * 10.0

        # Measurement Noise Covariance R
        self.R = np.eye(2, dtype=np.float32) * 2.0

        # Process Noise Covariance Q
        self.Q = np.eye(4, dtype=np.float32) * 0.1

        self.is_initialized = False

    def initialize_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        """Initializes Kalman state explicitly."""
        self.state[0, 0] = x
        self.state[1, 0] = y
        self.state[2, 0] = vx
        self.state[3, 0] = vy
        self.is_initialized = True
        self.missing_frames_count = 0

    def predict(self, dt: float = 0.033) -> Point2D:
        """Predicts ball position using velocity state dynamics."""
        if dt > 0:
            self.F[0, 2] = float(dt)
            self.F[1, 3] = float(dt)
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return Point2D(x=float(self.state[0, 0]), y=float(self.state[1, 0]))

    def update(
        self,
        detection: Union[BoundingBox, BallDetectionResult, np.ndarray, list, tuple, None],
        frame_data: Optional[FrameData] = None,
    ) -> BallTrackingFrameResult:
        """Updates Kalman filter state with new detection bounding box or detection result payload."""
        # Extract BoundingBox or raw (x, y) coordinates
        bbox: Optional[BoundingBox] = None
        raw_xy: Optional[tuple[float, float]] = None

        if isinstance(detection, BoundingBox):
            bbox = detection
        elif isinstance(detection, BallDetectionResult):
            if detection.has_ball and detection.detected_ball:
                bbox = detection.detected_ball.bbox
        elif isinstance(detection, (np.ndarray, list, tuple)) and len(detection) >= 2:
            raw_xy = (float(detection[0]), float(detection[1]))

        frame_num = frame_data.frame_number if frame_data else 0
        ts = frame_data.timestamp_seconds if frame_data else 0.0

        # 1. Prediction step
        predicted_pos = self.predict()

        if bbox is not None or raw_xy is not None:
            # Measurement available
            cx = bbox.center.x if bbox else raw_xy[0]
            cy = bbox.center.y if bbox else raw_xy[1]
            z = np.array([[cx], [cy]], dtype=np.float32)

            if not self.is_initialized:
                self.initialize_state(cx, cy, 0.0, 0.0)

            # Measurement Update Step
            y = z - np.dot(self.H, self.state)
            S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

            self.state = self.state + np.dot(K, y)
            self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

            self.missing_frames_count = 0
            is_interpolated = False
        else:
            # No measurement (Missing frame prediction)
            self.missing_frames_count += 1
            is_interpolated = True

        if not self.is_initialized or self.missing_frames_count > self.max_missing_frames:
            return BallTrackingFrameResult(frame_number=frame_num, timestamp_seconds=ts, tracked_ball=None)

        curr_x = float(self.state[0, 0])
        curr_y = float(self.state[1, 0])
        vx = float(self.state[2, 0])
        vy = float(self.state[3, 0])
        speed = math.sqrt(vx * vx + vy * vy)

        curr_pos = Point2D(x=curr_x, y=curr_y)
        self.trajectory_history.append(curr_pos)
        if len(self.trajectory_history) > 100:
            self.trajectory_history.pop(0)

        tracked = TrackedBall(
            center=curr_pos,
            velocity_x=vx,
            velocity_y=vy,
            speed_px_per_sec=speed,
            is_interpolated=is_interpolated,
            trajectory_history=list(self.trajectory_history),
        )

        return BallTrackingFrameResult(
            frame_number=frame_num,
            timestamp_seconds=ts,
            tracked_ball=tracked,
        )

    def reset(self) -> None:
        """Resets tracker state for new video stream."""
        self.is_initialized = False
        self.missing_frames_count = 0
        self.trajectory_history.clear()
        self.state.fill(0)


# Class alias for backward compatibility with early unit tests
BallKalmanFilter = BallTracker
