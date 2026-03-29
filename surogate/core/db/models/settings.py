"""Settings: ApiKey, Integration, NotificationPreference."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from surogate.core.db.base import Base, UUIDMixin, TimestampMixin


# ── Enums ──────────────────────────────────────────────────────────────


class ApiKeyStatus(enum.Enum):
    active = "active"
    revoked = "revoked"


class IntegrationStatus(enum.Enum):
    connected = "connected"
    disconnected = "disconnected"


# ── Models ─────────────────────────────────────────────────────────────


class ApiKey(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "api_keys"

    name: Mapped[str] = mapped_column(sa.String(255))
    prefix: Mapped[str] = mapped_column(sa.String(32))
    hashed_key: Mapped[str] = mapped_column(sa.String(512))
    scopes: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    status: Mapped[ApiKeyStatus] = mapped_column(
        sa.Enum(ApiKeyStatus), default=ApiKeyStatus.active
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    created_by: Mapped["User"] = relationship()  # noqa: F821


class Integration(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "integrations"

    name: Mapped[str] = mapped_column(sa.String(255))
    status: Mapped[IntegrationStatus] = mapped_column(
        sa.Enum(IntegrationStatus)
    )
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, onupdate=sa.func.now(), nullable=True
    )


class NotificationPreference(Base):
    __tablename__ = "notification_preferences"

    user_id: Mapped[str] = mapped_column(
        sa.ForeignKey("users.id"), primary_key=True
    )
    event: Mapped[str] = mapped_column(sa.String(255), primary_key=True)
    email: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    in_app: Mapped[bool] = mapped_column(sa.Boolean, default=True)


__all__ = [
    "ApiKeyStatus",
    "IntegrationStatus",
    "ApiKey",
    "Integration",
    "NotificationPreference",
]
