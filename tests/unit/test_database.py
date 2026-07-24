"""
Unit tests for Database Infrastructure (SQLAlchemy 2.0 Async ORM models & MatchRepository).
"""

import pytest
from infrastructure.database.connection import get_async_engine, get_async_session_factory
from infrastructure.database.models import Base
from infrastructure.database.repository import MatchRepository
from shared.domain.entities import TeamSide
from shared.schemas.events import EventType, MatchEvent


@pytest.fixture
async def async_session():
    # In-memory SQLite for async unit testing
    engine = get_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = get_async_session_factory(engine)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_match_repository_create_and_get(async_session):
    repo = MatchRepository(async_session)

    # Create match
    match = await repo.create_match(
        match_id="m_db_101",
        match_name="Champions League Final",
        home_team="REAL MADRID",
        away_team="BAYERN",
    )
    assert match.id == "m_db_101"
    assert match.home_score == 0

    # Fetch match
    fetched = await repo.get_match("m_db_101")
    assert fetched is not None
    assert fetched.match_name == "Champions League Final"


@pytest.mark.asyncio
async def test_match_repository_save_event(async_session):
    repo = MatchRepository(async_session)
    await repo.create_match("m_db_102", "Derby Match")

    event = MatchEvent(
        event_id="evt_db_101",
        match_id="m_db_102",
        event_type=EventType.GOAL,
        video_timestamp_seconds=24.5,
        frame_number=735,
        team=TeamSide.HOME,
        player_id=9,
        details={"speed": 82.5},
    )

    saved_event = await repo.save_event(event)
    assert saved_event.id == "evt_db_101"
    assert saved_event.event_type == "goal"

    # Fetch all events
    events = await repo.get_match_events("m_db_102")
    assert len(events) == 1
    assert events[0].details["speed"] == 82.5
