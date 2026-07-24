"""
Purpose: Scoreboard Engine & Match State Machine maintaining live match score, game timer, and event logging.
Dependencies: enum, datetime, shared.domain.entities, shared.schemas.events, shared.logging
Inputs: MatchEvent payloads, clock tick updates, and user control signals
Outputs: Live MatchState snapshot, formatted clock string, and event history
"""

from enum import Enum
from shared.domain.entities import MatchState, TeamSide
from shared.logging import setup_logger
from shared.schemas.events import EventType, MatchEvent

logger = setup_logger("scoreboard_engine", service_name="event_engine")


class MatchPeriod(str, Enum):
    NOT_STARTED = "not_started"
    FIRST_HALF = "1st_half"
    HALFTIME = "halftime"
    SECOND_HALF = "2nd_half"
    FULLTIME = "fulltime"
    PAUSED = "paused"


class ScoreboardEngine:
    """State Machine maintaining official match score, timer, period, and event log."""

    def __init__(
        self,
        match_id: str = "match_default",
        home_team_name: str = "HOME",
        away_team_name: str = "AWAY",
    ):
        self.match_id = match_id
        self.home_team_name = home_team_name
        self.away_team_name = away_team_name

        self.home_score = 0
        self.away_score = 0
        self.elapsed_seconds = 0.0
        self.period = MatchPeriod.NOT_STARTED
        self.event_log: list[MatchEvent] = []

    def start_match(self) -> None:
        """Starts match and transitions period to 1st Half."""
        self.period = MatchPeriod.FIRST_HALF
        logger.info(f"Match {self.match_id} started: 1st Half.")

    def pause_match(self) -> None:
        """Pauses game clock."""
        self.period = MatchPeriod.PAUSED
        logger.info(f"Match {self.match_id} paused.")

    def resume_match(self) -> None:
        """Resumes game clock."""
        if self.elapsed_seconds < 2700:  # 45 mins
            self.period = MatchPeriod.FIRST_HALF
        else:
            self.period = MatchPeriod.SECOND_HALF
        logger.info(f"Match {self.match_id} resumed.")

    def update_clock(self, dt: float) -> None:
        """Updates elapsed match clock by delta seconds if match is active."""
        if self.period in (MatchPeriod.FIRST_HALF, MatchPeriod.SECOND_HALF):
            self.elapsed_seconds += max(0.0, dt)

    def format_clock(self) -> str:
        """Formats elapsed match time into MM:SS string."""
        total_sec = int(self.elapsed_seconds)
        minutes = total_sec // 60
        seconds = total_sec % 60
        return f"{minutes:02d}:{seconds:02d}"

    def process_event(self, event: MatchEvent) -> None:
        """Consumes a MatchEvent to update scores and event history log."""
        self.event_log.append(event)

        if event.event_type == EventType.GOAL:
            if event.team == TeamSide.HOME:
                self.home_score += 1
                logger.info(f"⚽ SCORE UPDATE! {self.home_team_name} {self.home_score} - {self.away_score} {self.away_team_name}")
            elif event.team == TeamSide.AWAY:
                self.away_score += 1
                logger.info(f"⚽ SCORE UPDATE! {self.home_team_name} {self.home_score} - {self.away_score} {self.away_team_name}")

    def get_state_snapshot(self) -> MatchState:
        """Returns standard domain MatchState snapshot."""
        return MatchState(
            match_id=self.match_id,
            home_score=self.home_score,
            away_score=self.away_score,
            elapsed_seconds=round(self.elapsed_seconds, 1),
            is_active=self.period in (MatchPeriod.FIRST_HALF, MatchPeriod.SECOND_HALF),
        )

    def reset(self) -> None:
        """Resets scoreboard to initial 0-0 state."""
        self.home_score = 0
        self.away_score = 0
        self.elapsed_seconds = 0.0
        self.period = MatchPeriod.NOT_STARTED
        self.event_log.clear()
        logger.info("ScoreboardEngine state reset.")
