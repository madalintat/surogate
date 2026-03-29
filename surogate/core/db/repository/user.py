from pyparsing import Optional
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.platform import (
    User,
)

async def get_lakefs_credentials(
    session: AsyncSession, username: str
) -> Optional[tuple[str, str]]:
    """Return LakeFS credentials (key, secret) for a user."""
    result = await session.execute(
        sa.select(
            User.hub_key,
            User.hub_secret,
        ).where(User.username == username)
    )
    return result.one_or_none()