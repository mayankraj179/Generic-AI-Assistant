from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://assistant:assistant_dev_password@localhost:5432/assistant_framework",
)


def create_session_factory() -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(DATABASE_URL, future=True)
    return async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    factory = create_session_factory()
    async with factory() as session:
        yield session
