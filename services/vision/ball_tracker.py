"""
Purpose: Ball Tracking & Trajectory Estimation Engine using 2D Kalman Filtering.
Dependencies: numpy, math, services.vision.models, shared.domain.entities, shared.logging
Inputs: BallDetectionResult and FrameData
Outputs: BallTrackingFrameResult containing TrackedBall with position, velocity vector, and flight path
"""

import math
from typing import Optional
import numpy as np
from services.ingestion.frame import FrameData
from services.vision.models import BallDetectionResult, BallTrackingFrameResult, TrackedBall
from shared.domain.entities import Point2D
from shared.logging import setup_logger

logger = setup_logger("ball_tracker", service_name="vision")


class BallKalmanFilter:
    """2D Constant-Velocity Kalman Filter for tracking football position [x, y] and velocity [vx, vy]."""

    def __init__(self, process_noise_std: float = 1.0, measurement_noise_std: float = 2.0):
        # State vector: [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)

        # Covariance matrix P
        self.P = np.eye(4, dtype=np.float32) * 10.0

        # Measurement matrix H (we measure x and y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Measurement noise R
        self.R = np.eye(2, dtype=np.float32) * (measurement_noise_std ** 2)

        # Process noise Q
        self.q_std = process_noise_std
        self.last_timestamp: Optional[float] = None

    def predict(self, dt: float) -> np.ndarray:
        """Predicts state vector for delta time dt."""
        if dt <= 0:
            dt = 0.033  # Default 30 FPS step

        # State transition matrix F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        # Process noise matrix Q
        Q = np.eye(4, dtype=np.float32) * (self.q_std ** 2)

        # State prediction: x = F * x
        self.state = F @ self.state
        # Covariance prediction: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + Q

        return self.state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Updates Kalman Filter with measured [x, y] coordinates."""
        z = measurement.reshape(2, 1)

        # Innovation: y = z - H * x
        y = z - (self.H @ self.state)

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: x = x + K * y
        self.state = self.state + K @ y

        # Covariance update: P = (I - K * H) * P
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        return self.state

    def initialize_state(self, x: float, y: float) -> None:
        """Initializes state vector with starting position."""
        self.state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 5.0


class BallTracker:
    """Ball Tracker handling Kalman prediction, occlusion interpolation, and velocity vector calculation."""

    def __init__(self, max_missed_frames: int = 15, max_history_len: int = 30):
        self.max_missed_frames = max_missed_frames
        self.max_history_len = max_history_len

        self.kalman = BallKalmanFilter()
        self._is_initialized = False
        self._missed_frames = 0
        self._last_timestamp: Optional[float] = None
        self._trajectory_history: list[Point2D] = []

    def reset(self) -> None:
        """Resets tracking state and Kalman Filter."""
        self.kalman = BallKalmanFilter()
        self._is_initialized = False
        self._missed_frames = 0
        self._last_timestamp = None
        self._trajectory_history.clear()
        logger.info("BallTracker state reset.")

    def update(
        self, detection_result: BallDetectionResult, frame: FrameData
    ) -> BallTrackingFrameResult:
        """Updates ball tracker with YOLO ball detection or interpolates during occlusion."""
        dt = (
            frame.timestamp_seconds - self._last_timestamp
            if self._last_timestamp is not None
            else 1.0 / frame.metadata.fps
        )
        self._last_timestamp = frame.timestamp_seconds

        if dt <= 0:
            dt = 1.0 / frame.metadata.fps

        if detection_result.has_ball and detection_result.ball_object is not None:
            # Ball detected by YOLO
            meas_x = detection_result.ball_object.bbox.center.x
            meas_y = detection_result.ball_object.bbox.center.y
            confidence = detection_result.ball_object.confidence

            if not self._is_initialized:
                self.kalman.initialize_state(meas_x, meas_y)
                self._is_initialized = True
            else:
                self.kalman.predict(dt)
                self.kalman.update(np.array([meas_x, meas_y], dtype=np.float32))

            self._missed_frames = 0
            is_interpolated = False

        elif self._is_initialized and self._missed_frames < self.max_missed_frames:
            # Detection dropout (occlusion/blur) -> Predict using Kalman Filter
            self.kalman.predict(dt)
            self._missed_frames += 1
            is_interpolated = True
            confidence = max(0.2, 0.9 - (self._missed_frames * 0.05))

        else:
            # No ball detected and missed frames exceeded threshold -> Lost ball
            self._is_initialized = False
            return BallTrackingFrameResult(
                frame_number=frame.frame_number,
                timestamp_seconds=frame.timestamp_seconds,
                has_ball=False,
                tracked_ball=None,
            )

        # Extract predicted/updated state variables
        state = self.kalman.state
        center_x = float(state[0, 0])
        center_y = float(state[1, 0])
        vx = float(state[2, 0])
        vy = float(state[3, 0])

        speed_px_per_sec = round(math.sqrt(vx**2 + vy**2), 2)
        curr_center = Point2D(x=round(center_x, 2), y=round(center_y, 2))

        # Append to trajectory history
        self._trajectory_history.append(curr_center)
        if len(self._trajectory_history) > self.max_history_len:
            self._trajectory_history.pop(0)

        tracked_ball = TrackedBall(
            center=curr_center,
            velocity=Point2D(x=round(vx, 2), y=round(vy, 2)),
            speed_px_per_sec=speed_px_per_sec,
            is_interpolated=is_interpolated,
            confidence=round(confidence, 3),
            trajectory_history=list(self._trajectory_history),
        )

        return BallTrackingFrameResult(
            frame_number=frame.frame_number,
            timestamp_seconds=frame.timestamp_seconds,
            has_ball=True,
            tracked_ball=tracked_ball,
        )
