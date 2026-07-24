"""
Purpose: Goal Detection Event Engine evaluating spatial goal line checks and emitting validated MatchEvent payloads.
Dependencies: uuid, datetime, services.ingestion.frame, services.vision.models, shared.domain.entities, shared.schemas.events, shared.logging
Inputs: TrackedBall, GoalLineCheckResult, PlayerTrackingFrameResult, FrameData
Outputs: Optional[MatchEvent] when a legitimate goal is verified
"""

import uuid
from typing import Optional
from services.ingestion.frame import FrameData
from services.vision.models import GoalLineCheckResult, GoalSide, PlayerTrackingFrameResult, TrackedBall
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
        tracked_ball: Optional[TrackedBall],
        goal_check: GoalLineCheckResult,
        players_result: Optional[PlayerTrackingFrameResult],
        frame: FrameData,
    ) -> Optional[MatchEvent]:
        """Evaluates frame telemetry and returns a MatchEvent if a goal is verified."""
        if tracked_ball is None:
            return None

        current_time = frame.timestamp_seconds

        # Check cooldown lockout hysteresis
        if (
            self._last_goal_timestamp is not None
            and (current_time - self._last_goal_timestamp) < self.cooldown_seconds
        ):
            return None

        goal_side = goal_check.goal_side
        is_past_line = goal_check.is_ball_past_goal_line

        # Transition: Ball was outside net -> Now 100% past goal line plane into net
        if is_past_line and not self._in_goal_state[goal_side]:
            self._in_goal_state[goal_side] = True
            self._last_goal_timestamp = current_time

            # Determine scoring team (Home Goal side scored on -> Away team scored, and vice versa)
            scoring_team = TeamSide.AWAY if goal_side == GoalSide.HOME_GOAL else TeamSide.HOME

            # Find nearest player (attributed scorer)
            closest_player_id = None
            if players_result and players_result.tracked_players:
                ball_pt = tracked_ball.center
                min_dist = float("inf")
                for p in players_result.tracked_players:
                    dx = p.ground_position.x - ball_pt.x
                    dy = p.ground_position.y - ball_pt.y
                    dist = math.sqrt(dx * dx + dy * dy) if hasattr(math, 'sqrt') else (dx*dx + dy*dy)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_player_id = p.track_id

            event_id = f"evt_goal_{uuid.uuid4().hex[:8]}"
            goal_event = MatchEvent(
                event_id=event_id,
                match_id=self.match_id,
                event_type=EventType.GOAL,
                video_timestamp_seconds=round(current_time, 2),
                frame_number=frame.frame_number,
                team=scoring_team,
                player_id=closest_player_id,
                location=tracked_ball.center,
                details={
                    "goal_side": goal_side.value,
                    "ball_speed_px_sec": tracked_ball.speed_px_per_sec,
                    "signed_distance_meters": goal_check.signed_distance_meters,
                    "is_interpolated": tracked_ball.is_interpolated,
                },
            )

            logger.info(
                f"⚽ GOAL EVENT FIRED! [{scoring_team.value.upper()}] Frame: {frame.frame_number}, "
                f"Timestamp: {current_time:.2f}s, Goal Side: {goal_side.value}"
            )
            return goal_event

        elif not is_past_line:
            self._in_goal_state[goal_side] = False

        return None
