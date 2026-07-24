"""
Purpose: Goal Detection Event Engine evaluating spatial goal line checks and emitting validated MatchEvent payloads.
Dependencies: uuid, datetime, services.ingestion.frame, services.vision.models, shared.domain.entities, shared.schemas.events, shared.logging
Inputs: TrackedBall, GoalLineCheckResult, PlayerTrackingFrameResult, FrameData
Outputs: Optional[MatchEvent] when a legitimate goal is verified
"""

import math
import uuid
from typing import Optional, Union
from services.ingestion.frame import FrameData
from services.vision.models import GoalLineCheckResult, GoalSide, PlayerTrackingFrameResult, TrackedBall, TrackedPlayer
from shared.domain.entities import TeamSide
from shared.logging import setup_logger
from shared.schemas.events import EventType, MatchEvent

logger = setup_logger("goal_engine", service_name="event_engine")


class GoalEngine:
    """State machine event engine for goal verification and duplicate event lockout hysteresis."""

    def __init__(self, match_id: str = "match_default", cooldown_seconds: float = 5.0):
        self.match_id = match_id
        self.cooldown_seconds = cooldown_seconds

        self._last_goal_timestamp: Optional[float] = None
        self._in_goal_state: dict[GoalSide, bool] = {
            GoalSide.HOME_GOAL: False,
            GoalSide.AWAY_GOAL: False,
        }

    def reset(self) -> None:
        """Resets engine cooldown state."""
        self._last_goal_timestamp = None
        self._in_goal_state = {GoalSide.HOME_GOAL: False, GoalSide.AWAY_GOAL: False}
        logger.info("GoalEngine state reset.")

    def process_frame(
        self,
        tracked_ball: Optional[TrackedBall] = None,
        goal_check: Optional[GoalLineCheckResult] = None,
        players_result: Optional[Union[PlayerTrackingFrameResult, list[TrackedPlayer]]] = None,
        frame: Optional[FrameData] = None,
        **kwargs,
    ) -> Optional[MatchEvent]:
        """Evaluates frame telemetry and returns a MatchEvent if a goal is verified."""
        # Handle kwargs fallback for alternate parameter orders
        target_ball = tracked_ball if tracked_ball is not None else kwargs.get("tracked_ball")
        target_check = goal_check if goal_check is not None else kwargs.get("goal_check")
        target_frame = frame if frame is not None else kwargs.get("frame")
        target_players = players_result if players_result is not None else kwargs.get("tracked_players")

        if target_ball is None or target_check is None or target_frame is None:
            return None

        current_time = target_frame.timestamp_seconds

        # Check cooldown lockout hysteresis
        if (
            self._last_goal_timestamp is not None
            and (current_time - self._last_goal_timestamp) < self.cooldown_seconds
        ):
            return None

        goal_side = target_check.goal_side
        is_past_line = target_check.is_ball_past_goal_line

        # Transition: Ball was outside net -> Now 100% past goal line plane into net
        if is_past_line and not self._in_goal_state.get(goal_side, False):
            self._in_goal_state[goal_side] = True
            self._last_goal_timestamp = current_time

            # Determine scoring team (Home Goal side scored on -> Away team scored, and vice versa)
            scoring_team = TeamSide.AWAY if goal_side in (GoalSide.HOME_GOAL, GoalSide.LEFT) else TeamSide.HOME

            # Find nearest player (attributed scorer)
            closest_player_id = None
            player_list = (
                target_players.tracked_players
                if isinstance(target_players, PlayerTrackingFrameResult)
                else (target_players if isinstance(target_players, list) else [])
            )

            if player_list:
                ball_pt = target_ball.center
                min_dist = float("inf")
                for p in player_list:
                    dx = p.ground_position.x - ball_pt.x
                    dy = p.ground_position.y - ball_pt.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < min_dist:
                        min_dist = dist
                        closest_player_id = p.track_id

            event_id = f"evt_goal_{uuid.uuid4().hex[:8]}"
            goal_event = MatchEvent(
                event_id=event_id,
                match_id=self.match_id,
                event_type=EventType.GOAL,
                video_timestamp_seconds=round(current_time, 2),
                frame_number=target_frame.frame_number,
                team=scoring_team,
                player_id=closest_player_id,
                location=target_ball.center,
                details={
                    "goal_side": goal_side.value,
                    "ball_speed_px_sec": target_ball.speed_px_per_sec,
                    "signed_distance_meters": target_check.signed_distance_meters,
                    "is_interpolated": target_ball.is_interpolated,
                },
            )

            logger.info(
                f"⚽ GOAL EVENT FIRED! [{scoring_team.value.upper()}] Frame: {target_frame.frame_number}, "
                f"Timestamp: {current_time:.2f}s, Goal Side: {goal_side.value}"
            )
            return goal_event

        elif not is_past_line and goal_side in self._in_goal_state:
            self._in_goal_state[goal_side] = False

        return None
