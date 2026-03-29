"""SQLAlchemy 2.0 declarative base and common mixins."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UUIDMixin:
    """Adds a UUID string primary key."""

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )


class TimestampMixin:
    """Adds created_at with a server-side default."""

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
