"""Operate domain: Agent, Skill, Tool, McpServer, Model."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from surogate.core.db.base import Base, UUIDMixin, TimestampMixin


# ── Enums ──────────────────────────────────────────────────────────────


class AgentStatus(enum.Enum):
    running = "running"
    deploying = "deploying"
    stopped = "stopped"
    error = "error"


class AgentVersionStatus(enum.Enum):
    live = "live"
    previous = "previous"
    failed = "failed"


class SkillStatus(enum.Enum):
    active = "active"
    error = "error"


class ToolCategory(enum.Enum):
    tool = "tool"
    rag = "rag"
    workflow = "workflow"
    guardrail = "guardrail"
    output = "output"


class ToolStatus(enum.Enum):
    active = "active"
    error = "error"
    deploying = "deploying"


class McpTransport(enum.Enum):
    sse = "sse"
    stdio = "stdio"


class McpStatus(enum.Enum):
    connected = "connected"
    disconnected = "disconnected"
    error = "error"


class ModelType(enum.Enum):
    base = "base"
    fine_tuned = "fine-tuned"


class ModelStatus(enum.Enum):
    serving = "serving"
    stopped = "stopped"
    error = "error"


# ── Association tables ─────────────────────────────────────────────────

agent_skills = sa.Table(
    "agent_skills",
    Base.metadata,
    sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), primary_key=True),
    sa.Column("skill_id", sa.String(36), sa.ForeignKey("skills.id"), primary_key=True),
)

agent_tools = sa.Table(
    "agent_tools",
    Base.metadata,
    sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), primary_key=True),
    sa.Column("tool_id", sa.String(36), sa.ForeignKey("tools.id"), primary_key=True),
)

agent_mcp_servers = sa.Table(
    "agent_mcp_servers",
    Base.metadata,
    sa.Column("agent_id", sa.String(36), sa.ForeignKey("agents.id"), primary_key=True),
    sa.Column(
        "mcp_server_id",
        sa.String(36),
        sa.ForeignKey("mcp_servers.id"),
        primary_key=True,
    ),
)


# ── Models ─────────────────────────────────────────────────────────────


class Agent(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "agents"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    display_name: Mapped[str] = mapped_column(sa.String(255))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    version: Mapped[str] = mapped_column(sa.String(64))
    model_id: Mapped[str] = mapped_column(sa.ForeignKey("models.id"))
    status: Mapped[AgentStatus] = mapped_column(sa.Enum(AgentStatus))
    replicas: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    image: Mapped[str] = mapped_column(sa.String(512))
    endpoint: Mapped[str] = mapped_column(sa.String(2048))
    system_prompt: Mapped[str] = mapped_column(sa.Text, default="")
    env_vars: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    resources: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )

    project: Mapped["Project"] = relationship()  # noqa: F821
    model: Mapped[Model] = relationship()
    created_by: Mapped["User"] = relationship()  # noqa: F821
    versions: Mapped[list[AgentVersion]] = relationship(back_populates="agent")
    skills: Mapped[list[Skill]] = relationship(
        secondary=agent_skills, back_populates="agents"
    )
    tools: Mapped[list[Tool]] = relationship(
        secondary=agent_tools, back_populates="agents"
    )
    mcp_servers: Mapped[list[McpServer]] = relationship(
        secondary=agent_mcp_servers, back_populates="agents"
    )


class AgentVersion(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "agent_versions"

    agent_id: Mapped[str] = mapped_column(
        sa.ForeignKey("agents.id"), index=True
    )
    version: Mapped[str] = mapped_column(sa.String(64))
    git_hash: Mapped[str] = mapped_column(sa.String(40))
    status: Mapped[AgentVersionStatus] = mapped_column(sa.Enum(AgentVersionStatus))
    changelog: Mapped[str] = mapped_column(sa.Text, default="")
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    agent: Mapped[Agent] = relationship(back_populates="versions")
    created_by: Mapped["User"] = relationship()  # noqa: F821



class Skill(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "skills"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    display_name: Mapped[str] = mapped_column(sa.String(255))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    content: Mapped[str] = mapped_column(sa.Text, default="")
    license: Mapped[str] = mapped_column(sa.String(255))
    compatibility: Mapped[str] = mapped_column(sa.Text, default="")
    meta: Mapped[Optional[dict[str, str]]] = mapped_column("metadata", sa.JSON, nullable=True)
    allowed_tools: Mapped[Optional[list[str]]] = mapped_column(sa.JSON, nullable=True)
    status: Mapped[SkillStatus] = mapped_column(sa.Enum(SkillStatus))
    author_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))
    tags: Mapped[Optional[list[str]]] = mapped_column(sa.JSON, nullable=True)
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, onupdate=sa.func.now(), nullable=True
    )
    author: Mapped["User"] = relationship()  # noqa: F821
    agents: Mapped[list[Agent]] = relationship(
        secondary=agent_skills, back_populates="skills"
    )


class Tool(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "tools"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    display_name: Mapped[str] = mapped_column(sa.String(255))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    category: Mapped[ToolCategory] = mapped_column(sa.Enum(ToolCategory))
    version: Mapped[str] = mapped_column(sa.String(64))
    status: Mapped[ToolStatus] = mapped_column(sa.Enum(ToolStatus))
    input_schema: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    output_schema: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    config: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    author_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))
    tags: Mapped[Optional[list[str]]] = mapped_column(sa.JSON, nullable=True)
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, onupdate=sa.func.now(), nullable=True
    )

    author: Mapped["User"] = relationship()  # noqa: F821
    agents: Mapped[list[Agent]] = relationship(
        secondary=agent_tools, back_populates="tools"
    )


class McpServer(UUIDMixin, Base):
    __tablename__ = "mcp_servers"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    url: Mapped[str] = mapped_column(sa.String(2048))
    transport: Mapped[McpTransport] = mapped_column(sa.Enum(McpTransport))
    auth: Mapped[Optional[dict[str, Any]]] = mapped_column(sa.JSON, nullable=True)
    status: Mapped[McpStatus] = mapped_column(sa.Enum(McpStatus))
    latency: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    exposed_tools: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    description: Mapped[str] = mapped_column(sa.Text, default="")
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, onupdate=sa.func.now(), nullable=True
    )

    agents: Mapped[list[Agent]] = relationship(
        secondary=agent_mcp_servers, back_populates="mcp_servers"
    )


class Model(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "models"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    display_name: Mapped[str] = mapped_column(sa.String(255))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    family: Mapped[str] = mapped_column(sa.String(128))
    parameters: Mapped[str] = mapped_column(sa.String(32))
    type: Mapped[ModelType] = mapped_column(sa.Enum(ModelType))
    base_model_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    quantization: Mapped[Optional[str]] = mapped_column(
        sa.String(32), nullable=True
    )
    context_window: Mapped[int] = mapped_column(sa.Integer)
    status: Mapped[ModelStatus] = mapped_column(sa.Enum(ModelStatus))
    compute_target_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), nullable=True
    )
    serving_config: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    generation_defaults: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    gpu: Mapped[str] = mapped_column(sa.String(128), default="")
    vram: Mapped[str] = mapped_column(sa.String(64), default="")
    throughput: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    training_run_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), nullable=True
    )
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    deployed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    deployed_by_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("users.id"), nullable=True
    )

    deployed_by: Mapped[Optional["User"]] = relationship()  # noqa: F821


__all__ = [
    "AgentStatus",
    "AgentVersionStatus",
    "SkillStatus",
    "ToolCategory",
    "ToolStatus",
    "McpTransport",
    "McpStatus",
    "ModelType",
    "ModelStatus",
    "agent_skills",
    "agent_tools",
    "agent_mcp_servers",
    "Agent",
    "AgentVersion",
    "Skill",
    "Tool",
    "McpServer",
    "Model",
]
