"""
Purpose: Async Database Connection Factory for SQLAlchemy 2.0.
Dependencies: sqlalchemy.ext.asyncio, shared.config, shared.logging
Inputs: Database URL from environment configuration
Outputs: AsyncEngine and AsyncSession maker instances
"""

from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from shared.config import get_settings
from shared.logging import setup_logger

logger = setup_logger("database_connection", service_name="infrastructure")


def get_async_engine(db_url: str | None = None) -> AsyncEngine:
    """Creates a SQLAlchemy 2.0 AsyncEngine instance."""
    settings = get_settings()
    target_url = db_url or settings.DATABASE_URL
    logger.info(f"Initializing Async Database Engine for target: {target_url.split('@')[-1] if '@' in target_url else target_url}")
    return create_async_engine(target_url, echo=False, future=True)


def get_async_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Creates an async session maker bound to the provided AsyncEngine."""
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
