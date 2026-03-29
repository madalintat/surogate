import hashlib
import secrets

from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa

from sqlalchemy.ext.asyncio import AsyncSession
from surogate.core.db.models.platform import RefreshToken, User, UserRole, UserRoleAssignment
from surogate.utils.hashing import hash_password

async def get_user_and_secret(
    session: AsyncSession, subject: str
) -> Optional[tuple[str, str, str, bool]]:
    """Fetch (salt, password_hash, jwt_secret, must_change_password) for a user."""
    result = await session.execute(
        sa.select(
            User.salt,
            User.password_hash,
            User.jwt_secret,
            User.must_change_password,
        ).where(User.username == subject)
    )
    return result.one_or_none()

async def requires_password_change(
    session: AsyncSession, username: str
) -> bool:
    """Return whether the user must change the seeded default password."""
    result = await session.execute(
        sa.select(User.must_change_password).where(User.username == username)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return False
    return row


async def update_password(
    session: AsyncSession, username: str, new_password: str
) -> bool:
    """Update password, clear first-login requirement, rotate JWT secret."""
    salt, pwd_hash = hash_password(new_password)
    jwt_secret = secrets.token_urlsafe(64)

    result = await session.execute(
        sa.update(User)
        .where(User.username == username)
        .values(
            salt=salt,
            password_hash=pwd_hash,
            jwt_secret=jwt_secret,
            must_change_password=False,
        )
    )
    await session.commit()
    return result.rowcount > 0


async def revoke_user_refresh_tokens(session: AsyncSession, username: str) -> None:
    """Revoke all refresh tokens for a user (e.g. on logout)."""
    await session.execute(
        sa.delete(RefreshToken).where(RefreshToken.username == username)
    )
    await session.commit()



async def get_jwt_secret(username: str, session: AsyncSession) -> Optional[str]:
    """Return the current JWT signing secret for a user."""
    result = await session.execute(
        sa.select(User.jwt_secret).where(User.username == username)
    )
    return result.scalar_one_or_none()


async def save_refresh_token(token: str, username: str, expires_at: str, session: AsyncSession) -> None:
    """Save a refresh token in the database."""
    await session.execute(
        sa.insert(RefreshToken).values(
            token_hash=token,
            username=username,
            expires_at=expires_at
        )
    )
    await session.commit()


def _hash_token(token: str) -> str:
    """SHA-256 hash helper used for refresh token storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

async def verify_refresh_token(token: str, session: AsyncSession) -> Optional[str]:
    """
    Verify a refresh token and return the username.

    Returns the username if valid and not expired, None otherwise.
    The token is NOT consumed — it stays valid until it expires.
    """
    token_hash = _hash_token(token)

    # Clean up any expired tokens while we're here
    now = datetime.now(timezone.utc).isoformat()
    await session.execute(
        sa.delete(RefreshToken).where(RefreshToken.expires_at < now)
    )
    await session.commit()

    result = await session.execute(
        sa.select(RefreshToken.id, RefreshToken.username, RefreshToken.expires_at)
        .where(RefreshToken.token_hash == token_hash)
    )
    row = result.one_or_none()
    if row is None:
        return None

    # Check expiry
    expires_at = datetime.fromisoformat(row.expires_at)
    if datetime.now(timezone.utc) > expires_at:
        await session.execute(
            sa.delete(RefreshToken).where(RefreshToken.id == row.id)
        )
        await session.commit()
        return None

    return row.username


async def seed_admin_user(session: AsyncSession) -> None:
    """Create a default admin user if no users exist yet."""
    result = await session.execute(sa.select(sa.func.count()).select_from(User))
    if result.scalar_one() > 0:
        return

    salt, pwd_hash = hash_password("surogate")

    admin = User(
        username="surogate",
        name="Surogate Admin",
        salt=salt,
        password_hash=pwd_hash,
        jwt_secret=secrets.token_urlsafe(64),
        must_change_password=False,
        role_assignments=[UserRoleAssignment(role=UserRole.admin)],
    )
    session.add(admin)
    await session.commit()
