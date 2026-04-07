"""Model inference metrics – one lightweight row per proxied request."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from surogate.core.db.base import Base, UUIDMixin


class ModelMetric(UUIDMixin, Base):
    """One row per chat-completion request flowing through the proxy."""

    __tablename__ = "model_metrics"

    project_name: Mapped[str] = mapped_column(sa.String(255), index=True)
    run_name: Mapped[str] = mapped_column(sa.String(255), index=True)
    model: Mapped[str] = mapped_column(sa.String(255), index=True)

    # Tokens
    prompt_tokens: Mapped[int] = mapped_column(sa.Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(sa.Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(sa.Integer, default=0)

    # Performance
    latency_ms: Mapped[float] = mapped_column(sa.Float, default=0.0)

    # Result
    status_code: Mapped[int] = mapped_column(sa.Integer, default=200)
    is_streaming: Mapped[bool] = mapped_column(sa.Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), index=True,
    )

    __table_args__ = (
        sa.Index("ix_model_metrics_model_created", "model", "created_at"),
    )


__all__ = ["ModelMetric"]
