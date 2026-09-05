from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.storage.local import storage
from app.storage.base import StorageBackend


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding async SQLAlchemy session."""
    async for session in get_db():
        yield session


def get_storage_backend() -> StorageBackend:
    """FastAPI dependency yielding the configured StorageBackend."""
    return storage
