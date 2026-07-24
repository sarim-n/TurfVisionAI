"""
Purpose: Scoreboard Engine managing live match score, timer clock, match periods, and event history log.
Dependencies: pydantic, services.ingestion.frame, shared.domain.entities, shared.schemas.events, shared.logging
Inputs: MatchEvent objects and timer tick intervals
Outputs: MatchState domain snapshots and formatted match clock strings
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
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
    """Scoreboard state machine maintaining live score, game clock, periods, and event history."""

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
        self.current_period = MatchPeriod.NOT_STARTED
        self.is_active = False
        self.event_log: list[MatchEvent] = []

    @property
    def period(self) -> MatchPeriod:
        return self.current_period

    def start_match(self) -> None:
        """Starts match clock and sets active period to FIRST_HALF."""
        self.is_active = True
        self.current_period = MatchPeriod.FIRST_HALF
        logger.info(f"Match {self.match_id} started: 1st Half.")

    def pause_match(self) -> None:
        """Pauses game clock and sets period to PAUSED."""
        self.is_active = False
        self.current_period = MatchPeriod.PAUSED
        logger.info(f"Match {self.match_id} paused.")

    def update_clock(self, dt_seconds: float) -> float:
        """Ticks game timer forward by dt_seconds."""
        if self.is_active:
            self.elapsed_seconds += dt_seconds
        return self.elapsed_seconds

    # Alias for pipeline integration
    tick_clock = update_clock

    def process_event(self, event: MatchEvent) -> None:
        """Processes MatchEvent and updates score counters if event is a GOAL."""
        self.event_log.append(event)
        if event.event_type == EventType.GOAL:
            if event.team == TeamSide.HOME:
                self.home_score += 1
            elif event.team == TeamSide.AWAY:
                self.away_score += 1
            logger.info(f"Score updated! {self.home_team_name} {self.home_score} - {self.away_score} {self.away_team_name}")

    def format_clock(self) -> str:
        """Formats elapsed game time into MM:SS string."""
        total_sec = int(self.elapsed_seconds)
        minutes = total_sec // 60
        seconds = total_sec % 60
        return f"{minutes:02d}:{seconds:02d}"

    def get_state_snapshot(self) -> MatchState:
        """Generates immutable MatchState snapshot."""
        return MatchState(
            match_id=self.match_id,
            home_score=self.home_score,
            away_score=self.away_score,
            elapsed_seconds=self.elapsed_seconds,
            current_period=self.current_period.value,
            is_active=self.is_active,
        )
