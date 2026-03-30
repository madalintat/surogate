from typing import Optional
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.platform import (
    User,
)

async def get_lakefs_credentials(session: AsyncSession, username: str) -> Optional[tuple[str, str]]:
    """Return LakeFS credentials (key, secret) for a user."""
    result = await session.execute(
        sa.select(
            User.hub_key,
            User.hub_secret,
        ).where(User.username == username)
    )
    return result.one_or_none()

async def set_lakefs_credentials(
    session: AsyncSession, username: str, key: str, secret: str
) -> None:
    """Set LakeFS credentials (key, secret) for a user."""
    await session.execute(
        sa.update(User)
        .where(User.username == username)
        .values(hub_key=key, hub_secret=secret)
    )
    await session.commit()