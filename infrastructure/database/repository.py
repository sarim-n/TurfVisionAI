"""
Purpose: Match Repository enforcing Clean Architecture data access abstractions.
Dependencies: sqlalchemy.ext.asyncio, infrastructure.database.models, shared.schemas.events, shared.logging
Inputs: AsyncSession database transactions
Outputs: Persisted ORM Entities and Query Result Lists
"""

from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from infrastructure.database.models import MatchEventModel, MatchModel
from shared.logging import setup_logger
from shared.schemas.events import MatchEvent

logger = setup_logger("match_repository", service_name="infrastructure")


class MatchRepository:
    """Async Repository managing Match and MatchEvent database transactions."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_match(
        self,
        match_id: str,
        match_name: str,
        home_team: str = "HOME",
        away_team: str = "AWAY",
    ) -> MatchModel:
        """Creates and persists a new MatchModel entity."""
        match_entity = MatchModel(
            id=match_id,
            match_name=match_name,
            home_team=home_team,
            away_team=away_team,
            home_score=0,
            away_score=0,
        )
        self.session.add(match_entity)
        await self.session.commit()
        await self.session.refresh(match_entity)
        logger.info(f"Persisted Match {match_id} to database.")
        return match_entity

    async def get_match(self, match_id: str) -> Optional[MatchModel]:
        """Fetches a MatchModel entity by match_id."""
        stmt = select(MatchModel).where(MatchModel.id == match_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_match_score(self, match_id: str, home_score: int, away_score: int) -> None:
        """Updates home and away score counters for a match."""
        match_entity = await self.get_match(match_id)
        if match_entity:
            match_entity.home_score = home_score
            match_entity.away_score = away_score
            await self.session.commit()
            logger.info(f"Updated Match {match_id} scores: {home_score} - {away_score}")

    async def save_event(self, event: MatchEvent) -> MatchEventModel:
        """Persists a MatchEvent entity."""
        event_entity = MatchEventModel(
            id=event.event_id,
            match_id=event.match_id,
            event_type=event.event_type.value,
            timestamp_seconds=event.video_timestamp_seconds,
            frame_number=event.frame_number,
            team=event.team.value,
            player_id=event.player_id,
            details=event.details,
        )
        self.session.add(event_entity)
        await self.session.commit()
        await self.session.refresh(event_entity)
        logger.info(f"Persisted MatchEvent {event.event_id} ({event.event_type.value}) for match {event.match_id}.")
        return event_entity

    async def get_match_events(self, match_id: str) -> list[MatchEventModel]:
        """Returns all persisted MatchEvent entities for a match."""
        stmt = select(MatchEventModel).where(MatchEventModel.match_id == match_id).order_by(MatchEventModel.frame_number)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
