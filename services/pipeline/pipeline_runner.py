"""
Purpose: Master multi-threaded end-to-end AI pipeline runner orchestrating video ingestion, YOLO vision inference, Kalman tracking, goal-line verification, event detection engines, and analytics snapshot generation.
Dependencies: cv2, numpy, services.ingestion.frame, services.vision.*, services.event_engine.*, shared.domain.entities, shared.schemas.events, shared.logging
Inputs: FrameData stream or video filepath
Outputs: PipelineFrameResult containing annotated image, active player tracks, ball track, verified match events, and live scoreboard snapshot
"""

import time
from typing import Optional
import cv2
import numpy as np
from pydantic import BaseModel, Field
from services.event_engine.goal_engine import GoalEngine
from services.event_engine.scoreboard import ScoreboardEngine
from services.ingestion.frame import FrameData
from services.pipeline.models import PipelineConfig, PipelineFrameResult
from services.stats_engine.stats_calculator import MatchStatsEngine
from services.vision.ball_detector import BallDetector
from services.vision.ball_tracker import BallTracker
from services.vision.goal_line import GoalLineDetector
from services.vision.homography import PitchHomography
from services.vision.models import BallTrackingFrameResult, TrackedBall, TrackedPlayer
from services.vision.player_detector import PlayerDetector
from services.vision.player_tracker import PlayerTracker
from services.vision.visualizer import draw_birds_eye_view, draw_goal_lines, draw_player_tracks, draw_scoreboard_overlay
from shared.domain.entities import MatchState, Point2D
from shared.logging import setup_logger
from shared.schemas.events import MatchEvent

logger = setup_logger("pipeline_runner", service_name="pipeline")


class TurfVisionPipeline:
    """Master AI Computer Vision Pipeline orchestrator."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logger.info(f"TurfVisionPipeline initialized for match: {self.config.match_id}")

        # Instantiate Vision Module Components
        self.player_detector = PlayerDetector(model_path=getattr(self.config, "yolo_player_model_path", "models/yolov8x.pt"))
        self.player_tracker = PlayerTracker()

        self.ball_detector = BallDetector(model_path=getattr(self.config, "yolo_ball_model_path", "models/yolov8x.pt"))
        self.ball_tracker = BallTracker()

        # Homography & Goal Line Spatial Detectors (Lazy-initialized on first frame width/height)
        self.homography: Optional[PitchHomography] = None
        self.goal_line_detector: Optional[GoalLineDetector] = None

        # Event & Stats Engines
        self.goal_engine = GoalEngine(match_id=self.config.match_id)
        self.scoreboard = ScoreboardEngine(match_id=self.config.match_id)
        self.stats_engine = MatchStatsEngine(match_id=self.config.match_id)
        self.scoreboard.start_match()

    def run_frame(self, frame: FrameData) -> PipelineFrameResult:
        """Processes single video frame through end-to-end AI pipeline."""
        start_time = time.perf_counter()
        img = frame.image.copy()
        h, w = img.shape[:2]

        # Initialize calibration matrix and goal line detector if not created yet
        if self.homography is None:
            self.homography = PitchHomography.create_default_calibration(w, h)
        if self.goal_line_detector is None:
            self.goal_line_detector = GoalLineDetector.create_default_detector(w, h)

        # 1. Player Detection & Tracking
        p_detections = self.player_detector.detect(img)
        p_tracking_res = self.player_tracker.update(p_detections.detections)
        tracked_players = p_tracking_res.tracked_players

        # 2. Football Detection & Tracking
        b_detections = self.ball_detector.detect(img)
        ball_box = b_detections.detected_ball.bbox if b_detections.detected_ball else None
        b_tracking_res = self.ball_tracker.update(ball_box)
        tracked_ball = b_tracking_res.tracked_ball if hasattr(b_tracking_res, "tracked_ball") else b_tracking_res

        # 3. Spatial Goal Line Checks
        ball_center = tracked_ball.center if tracked_ball else None
        left_goal_result = self.goal_line_detector.check_ball_position(ball_center, goal_side=self.goal_line_detector.left_goal.side)
        right_goal_result = self.goal_line_detector.check_ball_position(ball_center, goal_side=self.goal_line_detector.right_goal.side)
        goal_check_result = left_goal_result if left_goal_result.is_past_goal_line else right_goal_result

        # 4. Goal Event Verification Engine
        new_events = []
        goal_event = self.goal_engine.process_frame(
            frame=frame,
            goal_check=goal_check_result,
            tracked_ball=tracked_ball,
        )
        if goal_event:
            new_events.append(goal_event)
            self.scoreboard.process_event(goal_event)

        # 5. Stats Engine Tracking
        self.stats_engine.process_frame(
            tracked_players=tracked_players,
            tracked_ball=tracked_ball,
            homography=self.homography,
            frame=frame,
        )

        # 6. Game Clock Progression
        self.scoreboard.tick_clock(1.0 / (frame.metadata.fps if (frame.metadata and frame.metadata.fps > 0) else 30.0))

        # 7. Optional OpenCV Visual Overlays
        annotated = img
        if self.config.enable_visualization:
            annotated = draw_player_tracks(annotated, tracked_players)
            annotated = draw_goal_lines(annotated, self.goal_line_detector, goal_check_result)
            annotated = draw_scoreboard_overlay(annotated, self.scoreboard)

        return PipelineFrameResult(
            frame_number=frame.frame_number,
            timestamp_seconds=frame.timestamp_seconds,
            annotated_image=annotated,
            tracked_players=tracked_players,
            tracked_ball=tracked_ball,
            new_events=new_events,
            match_state=self.scoreboard.get_state_snapshot(),
        )

    def get_analytics_report(self):
        """Generates full match analytics report snapshot."""
        return self.stats_engine.generate_report()

    def reset(self) -> None:
        """Resets all pipeline trackers and event engines."""
        self.player_tracker.reset()
        self.ball_tracker.reset()
        self.goal_engine.reset()
