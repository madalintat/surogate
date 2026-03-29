"""Train domain: Dataset, SFT, RL, AgentFlow, Evaluator, Checkpoint."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from surogate.core.db.base import Base, UUIDMixin, TimestampMixin


# ── Enums ──────────────────────────────────────────────────────────────


class DatasetFormat(enum.Enum):
    sft = "sft"
    dpo = "dpo"
    grpo = "grpo"
    rlhf = "rlhf"


class DatasetStatus(enum.Enum):
    ready = "ready"
    building = "building"
    error = "error"


class PipelineStepType(enum.Enum):
    source = "source"
    transform = "transform"
    output = "output"


class PipelineStepStatus(enum.Enum):
    completed = "completed"
    running = "running"
    pending = "pending"


class ExperimentStatus(enum.Enum):
    active = "active"
    completed = "completed"
    archived = "archived"


class SFTMethod(enum.Enum):
    sft = "SFT"
    dpo = "DPO"


class RLMethod(enum.Enum):
    grpo = "GRPO"
    ppo = "PPO"


class RunStatus(enum.Enum):
    running = "running"
    completed = "completed"
    queued = "queued"
    failed = "failed"


class ComputeTarget(enum.Enum):
    local = "local"
    aws = "aws"
    gcp = "gcp"


class AgentFlowType(enum.Enum):
    single_agent = "single-agent"
    multi_agent = "multi-agent"


class AgentFlowStatus(enum.Enum):
    active = "active"
    error = "error"


class RewardType(enum.Enum):
    binary = "binary"
    continuous = "continuous"


class EvaluatorStatus(enum.Enum):
    active = "active"
    error = "error"


# ── Dataset models ─────────────────────────────────────────────────────


class Dataset(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "datasets"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    display_name: Mapped[str] = mapped_column(sa.String(255))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    format: Mapped[DatasetFormat] = mapped_column(sa.Enum(DatasetFormat))
    source: Mapped[str] = mapped_column(sa.String(512), default="")
    status: Mapped[DatasetStatus] = mapped_column(sa.Enum(DatasetStatus))
    samples: Mapped[int] = mapped_column(sa.Integer, default=0)
    tokens: Mapped[int] = mapped_column(sa.Integer, default=0)
    size: Mapped[str] = mapped_column(sa.String(32), default="")
    tags: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    published: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, onupdate=sa.func.now(), nullable=True
    )
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    versions: Mapped[list[DatasetVersion]] = relationship(back_populates="dataset")
    pipeline_steps: Mapped[list[DatasetPipelineStep]] = relationship(
        back_populates="dataset"
    )
    created_by: Mapped["User"] = relationship()  # noqa: F821


class DatasetVersion(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "dataset_versions"

    dataset_id: Mapped[str] = mapped_column(
        sa.ForeignKey("datasets.id"), index=True
    )
    version: Mapped[str] = mapped_column(sa.String(64))
    samples: Mapped[int] = mapped_column(sa.Integer, default=0)
    changelog: Mapped[str] = mapped_column(sa.Text, default="")
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    dataset: Mapped[Dataset] = relationship(back_populates="versions")
    created_by: Mapped["User"] = relationship()  # noqa: F821


class DatasetPipelineStep(UUIDMixin, Base):
    __tablename__ = "dataset_pipeline_steps"

    dataset_id: Mapped[str] = mapped_column(
        sa.ForeignKey("datasets.id"), index=True
    )
    order: Mapped[int] = mapped_column(sa.Integer)
    name: Mapped[str] = mapped_column(sa.String(255))
    type: Mapped[PipelineStepType] = mapped_column(sa.Enum(PipelineStepType))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    status: Mapped[PipelineStepStatus] = mapped_column(
        sa.Enum(PipelineStepStatus)
    )
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )

    dataset: Mapped[Dataset] = relationship(back_populates="pipeline_steps")


# ── SFT models ─────────────────────────────────────────────────────────


class SFTExperiment(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "sft_experiments"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    hypothesis: Mapped[str] = mapped_column(sa.Text, default="")
    status: Mapped[ExperimentStatus] = mapped_column(sa.Enum(ExperimentStatus))
    base_model: Mapped[str] = mapped_column(sa.String(255))
    tags: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    runs: Mapped[list[SFTRun]] = relationship(back_populates="experiment")
    created_by: Mapped["User"] = relationship()  # noqa: F821


class SFTRun(UUIDMixin, Base):
    __tablename__ = "sft_runs"

    experiment_id: Mapped[str] = mapped_column(
        sa.ForeignKey("sft_experiments.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    method: Mapped[SFTMethod] = mapped_column(sa.Enum(SFTMethod))
    status: Mapped[RunStatus] = mapped_column(sa.Enum(RunStatus))
    dataset_id: Mapped[str] = mapped_column(sa.ForeignKey("datasets.id"))
    model_name: Mapped[str] = mapped_column(sa.String(255))
    base_model: Mapped[str] = mapped_column(sa.String(255))
    progress: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    # Hyperparameters
    lr: Mapped[float] = mapped_column(sa.Float)
    batch_size: Mapped[int] = mapped_column(sa.Integer)
    grad_accum: Mapped[int] = mapped_column(sa.Integer, default=1)
    epochs: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    steps: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    warmup_steps: Mapped[int] = mapped_column(sa.Integer, default=0)
    weight_decay: Mapped[float] = mapped_column(sa.Float, default=0.0)
    scheduler: Mapped[str] = mapped_column(sa.String(64), default="cosine")
    optimizer: Mapped[str] = mapped_column(sa.String(64), default="AdamW")
    lora: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    # DPO-specific
    beta: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    label_smoothing: Mapped[Optional[float]] = mapped_column(
        sa.Float, nullable=True
    )
    ref_model: Mapped[Optional[str]] = mapped_column(
        sa.String(255), nullable=True
    )
    # Metrics / curves
    best_loss: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    loss_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    lr_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    grad_norm_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    eval_results: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    # Compute
    compute: Mapped[ComputeTarget] = mapped_column(
        sa.Enum(ComputeTarget), default=ComputeTarget.local
    )
    gpu: Mapped[str] = mapped_column(sa.String(128), default="")
    gpu_util: Mapped[float] = mapped_column(sa.Float, default=0.0)
    started_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    duration: Mapped[Optional[str]] = mapped_column(sa.String(64), nullable=True)
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )

    experiment: Mapped[SFTExperiment] = relationship(back_populates="runs")
    dataset: Mapped["Dataset"] = relationship()
    checkpoints: Mapped[list[Checkpoint]] = relationship(
        back_populates="sft_run",
        foreign_keys="[Checkpoint.sft_run_id]",
    )


# ── RL models ──────────────────────────────────────────────────────────


class AgentFlow(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "agent_flows"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    type: Mapped[AgentFlowType] = mapped_column(sa.Enum(AgentFlowType))
    agents: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    max_rollout_tokens: Mapped[int] = mapped_column(sa.Integer)
    steps_per_episode: Mapped[str] = mapped_column(sa.String(32), default="")
    source: Mapped[str] = mapped_column(sa.String(512), default="")
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    status: Mapped[AgentFlowStatus] = mapped_column(sa.Enum(AgentFlowStatus))
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    created_by: Mapped["User"] = relationship()  # noqa: F821


class Evaluator(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "evaluators"

    project_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("projects.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    description: Mapped[str] = mapped_column(sa.Text, default="")
    reward_type: Mapped[RewardType] = mapped_column(sa.Enum(RewardType))
    signals: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    built_in: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    status: Mapped[EvaluatorStatus] = mapped_column(sa.Enum(EvaluatorStatus))
    created_by_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("users.id"), nullable=True
    )

    created_by: Mapped[Optional["User"]] = relationship()  # noqa: F821


class RLExperiment(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "rl_experiments"

    project_id: Mapped[str] = mapped_column(
        sa.ForeignKey("projects.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    hypothesis: Mapped[str] = mapped_column(sa.Text, default="")
    status: Mapped[ExperimentStatus] = mapped_column(sa.Enum(ExperimentStatus))
    base_model: Mapped[str] = mapped_column(sa.String(255))
    tags: Mapped[Optional[list[Any]]] = mapped_column(sa.JSON, nullable=True)
    created_by_id: Mapped[str] = mapped_column(sa.ForeignKey("users.id"))

    runs: Mapped[list[RLRun]] = relationship(back_populates="experiment")
    created_by: Mapped["User"] = relationship()  # noqa: F821


class RLRun(UUIDMixin, Base):
    __tablename__ = "rl_runs"

    experiment_id: Mapped[str] = mapped_column(
        sa.ForeignKey("rl_experiments.id"), index=True
    )
    name: Mapped[str] = mapped_column(sa.String(255))
    method: Mapped[RLMethod] = mapped_column(sa.Enum(RLMethod))
    status: Mapped[RunStatus] = mapped_column(sa.Enum(RunStatus))
    agent_flow_id: Mapped[str] = mapped_column(sa.ForeignKey("agent_flows.id"))
    evaluator_id: Mapped[str] = mapped_column(sa.ForeignKey("evaluators.id"))
    dataset_id: Mapped[str] = mapped_column(sa.ForeignKey("datasets.id"))
    model_name: Mapped[str] = mapped_column(sa.String(255))
    base_model: Mapped[str] = mapped_column(sa.String(255))
    progress: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    # Algorithm config
    algorithm: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    workflow: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    # Hyperparameters
    lr: Mapped[float] = mapped_column(sa.Float)
    batch_size: Mapped[int] = mapped_column(sa.Integer)
    grad_accum: Mapped[int] = mapped_column(sa.Integer, default=1)
    epochs: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    steps: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    warmup_steps: Mapped[int] = mapped_column(sa.Integer, default=0)
    weight_decay: Mapped[float] = mapped_column(sa.Float, default=0.0)
    scheduler: Mapped[str] = mapped_column(sa.String(64), default="cosine")
    optimizer: Mapped[str] = mapped_column(sa.String(64), default="AdamW")
    lora: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    # Episode metrics
    episodes: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    avg_steps_per_episode: Mapped[float] = mapped_column(sa.Float, default=0.0)
    # Training curves
    best_loss: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    loss_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    reward_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    kl_div_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    policy_loss_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    grad_norm_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    lr_curve: Mapped[Optional[list[Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    eval_results: Mapped[Optional[dict[str, Any]]] = mapped_column(
        sa.JSON, nullable=True
    )
    # Compute
    compute: Mapped[ComputeTarget] = mapped_column(
        sa.Enum(ComputeTarget), default=ComputeTarget.local
    )
    gpu: Mapped[str] = mapped_column(sa.String(128), default="")
    gpu_util: Mapped[float] = mapped_column(sa.Float, default=0.0)
    started_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True
    )
    duration: Mapped[Optional[str]] = mapped_column(sa.String(64), nullable=True)
    hub_ref: Mapped[Optional[str]] = mapped_column(
        sa.String(512), nullable=True
    )

    experiment: Mapped[RLExperiment] = relationship(back_populates="runs")
    agent_flow: Mapped[AgentFlow] = relationship()
    evaluator: Mapped[Evaluator] = relationship()
    dataset: Mapped[Dataset] = relationship()
    checkpoints: Mapped[list[Checkpoint]] = relationship(
        back_populates="rl_run",
        foreign_keys="[Checkpoint.rl_run_id]",
    )


# ── Checkpoint (shared by SFT and RL) ─────────────────────────────────


class Checkpoint(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "checkpoints"

    sft_run_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("sft_runs.id"), nullable=True, index=True
    )
    rl_run_id: Mapped[Optional[str]] = mapped_column(
        sa.ForeignKey("rl_runs.id"), nullable=True, index=True
    )
    step: Mapped[int] = mapped_column(sa.Integer)
    loss: Mapped[float] = mapped_column(sa.Float)
    reward: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)
    best: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    path: Mapped[str] = mapped_column(sa.String(512))

    sft_run: Mapped[Optional[SFTRun]] = relationship(
        back_populates="checkpoints", foreign_keys=[sft_run_id]
    )
    rl_run: Mapped[Optional[RLRun]] = relationship(
        back_populates="checkpoints", foreign_keys=[rl_run_id]
    )


__all__ = [
    "DatasetFormat",
    "DatasetStatus",
    "PipelineStepType",
    "PipelineStepStatus",
    "ExperimentStatus",
    "SFTMethod",
    "RLMethod",
    "RunStatus",
    "ComputeTarget",
    "AgentFlowType",
    "AgentFlowStatus",
    "RewardType",
    "EvaluatorStatus",
    "Dataset",
    "DatasetVersion",
    "DatasetPipelineStep",
    "SFTExperiment",
    "SFTRun",
    "AgentFlow",
    "Evaluator",
    "RLExperiment",
    "RLRun",
    "Checkpoint",
]
