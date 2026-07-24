"""
Purpose: Declarative SQLAlchemy 2.0 ORM Models for TurfVision AI Database.
Dependencies: sqlalchemy, datetime, shared.domain.entities
Inputs: Data definitions
Outputs: SQLAlchemy ORM Mapped Classes
"""

from datetime import datetime, timezone
from typing import Any
from sqlalchemy import JSON, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base Declarative Class for ORM models."""
    pass


class MatchModel(Base):
    """SQLAlchemy ORM Model for Football Matches."""
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    match_name: Mapped[str] = mapped_column(String, nullable=False)
    home_team: Mapped[str] = mapped_column(String, nullable=False, default="HOME")
    away_team: Mapped[str] = mapped_column(String, nullable=False, default="AWAY")
    home_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    away_score: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(timezone.utc).isoformat()
    )


class MatchEventModel(Base):
    """SQLAlchemy ORM Model for Match Events (Goals, Shots, Passes)."""
    __tablename__ = "match_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    match_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    team: Mapped[str] = mapped_column(String, nullable=False)
    player_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
