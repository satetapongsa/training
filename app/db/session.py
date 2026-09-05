from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from app.core.logging import logger
from app.db.base import Base

# Async Engine (For FastAPI async endpoints)
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Synchronous Engine (For background training workers and CLI)
sync_db_url = settings.DATABASE_URL.replace("sqlite+aiosqlite://", "sqlite://")
if "postgresql+asyncpg://" in settings.DATABASE_URL:
    sync_db_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

sync_engine = create_engine(sync_db_url, echo=False)
SyncSessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


async def init_db() -> None:
    """Initializes the database schema asynchronously."""
    logger.info("Initializing database tables...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")


def init_db_sync() -> None:
    """Initializes the database schema synchronously."""
    Base.metadata.create_all(bind=sync_engine)



async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for injecting async DB sessions into FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
