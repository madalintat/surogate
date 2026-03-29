"""Observe domain: Conversation, Message, ToolCall, Annotation, Alert."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from surogate.core.db.base import Base, UUIDMixin, TimestampMixin


# ── Enums ──────────────────────────────────────────────────────────────


class ConversationStatus(enum.Enum):
    active = "active"
    completed = "completed"
    escalated = "escalated"


class Sentiment(enum.Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class MessageRole(enum.Enum):
    user = "user"
    assistant = "assistant"


class ToolCallStatus(enum.Enum):
    success = "success"
    error = "error"
    timeout = "timeout"


class AnnotationType(enum.Enum):
    skill_gap = "skill_gap"
    trajectory_correction = "trajectory_correction"
    quality_issue = "quality_issue"


class AlertSeverity(enum.Enum):
    critical = "critical"
    warning = "warning"
    info = "info"


class AlertSourceType(enum.Enum):
    agent = "agent"
    model = "model"
    compute = "compute"
    policy = "policy"


# ── Models ─────────────────────────────────────────────────────────────


class Conversation(UUIDMixin, Base):
    __tablename__ = "conversations"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    agent_id: Mapped[str] = mapped_column(sa.ForeignKey("agents.id"), index=True)
    user_id: Mapped[str] = mapped_column(sa.String(255))
    status: Mapped[ConversationStatus] = mapped_column(
        sa.Enum(ConversationStatus)
    )
    sentiment: Mapped[Optional[Sentiment]] = mapped_column(
        sa.Enum(Sentiment), nullable=True
    )
    sentiment_score: Mapped[Optional[float]] = mapped_column(
        sa.Float, nullable=True
    )
    flagged: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    starred: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    resolved: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    escalated: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    total_tokens: Mapped[int] = mapped_column(sa.Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(sa.Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(sa.Integer, default=0)
    turn_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    duration: Mapped[str] = mapped_column(sa.String(64), default="")
    avg_latency: Mapped[str] = mapped_column(sa.String(64), default="")
    tags: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    dataset_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("datasets.id"), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )

    messages: Mapped[list[Message]] = relationship(back_populates="conversation")


class Message(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(
        sa.ForeignKey("conversations.id"), index=True
    )
    role: Mapped[MessageRole] = mapped_column(sa.Enum(MessageRole))
    content: Mapped[str] = mapped_column(sa.Text)
    tokens: Mapped[int] = mapped_column(sa.Integer, default=0)
    latency: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    tool_calls: Mapped[list[ToolCall]] = relationship(back_populates="message")
    annotations: Mapped[list[Annotation]] = relationship(back_populates="message")


class ToolCall(UUIDMixin, Base):
    __tablename__ = "tool_calls"

    message_id: Mapped[str] = mapped_column(
        sa.ForeignKey("messages.id"), index=True
    )
    tool_name: Mapped[str] = mapped_column(sa.String(255))
    action: Mapped[str] = mapped_column(sa.String(255))
    status: Mapped[ToolCallStatus] = mapped_column(sa.Enum(ToolCallStatus))
    latency: Mapped[int] = mapped_column(sa.Integer, default=0)
    input: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    output: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )

    message: Mapped[Message] = relationship(back_populates="tool_calls")


class Annotation(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "annotations"

    message_id: Mapped[str] = mapped_column(
        sa.ForeignKey("messages.id"), index=True
    )
    type: Mapped[AnnotationType] = mapped_column(sa.Enum(AnnotationType))
    note: Mapped[str] = mapped_column(sa.Text)
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    message: Mapped[Message] = relationship(back_populates="annotations")
    created_by: Mapped["User"] = relationship()  # noqa: F821


class Alert(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "alerts"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    severity: Mapped[AlertSeverity] = mapped_column(sa.Enum(AlertSeverity))
    title: Mapped[str] = mapped_column(sa.String(255))
    message: Mapped[str] = mapped_column(sa.Text)
    acknowledged: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    acknowledged_by_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("users.id"), nullable=True
    )
    source_type: Mapped[AlertSourceType] = mapped_column(sa.Enum(AlertSourceType))
    source_id: Mapped[str] = mapped_column(sa.String(36))

    acknowledged_by: Mapped[Optional["User"]] = relationship()  # noqa: F821


__all__ = [
    "ConversationStatus",
    "Sentiment",
    "MessageRole",
    "ToolCallStatus",
    "AnnotationType",
    "AlertSeverity",
    "AlertSourceType",
    "Conversation",
    "Message",
    "ToolCall",
    "Annotation",
    "Alert",
]
