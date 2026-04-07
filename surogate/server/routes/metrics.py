"""Model metrics API – time-series aggregation over model_metrics rows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, cast, func, select, Float
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.engine import get_session
from surogate.core.db.models.metrics import ModelMetric
from surogate.server.auth.authentication import get_current_subject

router = APIRouter()

# Supported aggregation periods → SQL date_trunc keys
_PERIODS = {"minute", "hour", "day", "week", "month", "year"}

# Default lookback per period (if no explicit start/end given)
_DEFAULT_RANGE: dict[str, timedelta] = {
    "minute": timedelta(hours=1),
    "hour": timedelta(hours=24),
    "day": timedelta(days=30),
    "week": timedelta(days=90),
    "month": timedelta(days=365),
    "year": timedelta(days=365 * 3),
}


@router.get("/")
async def get_metrics(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    model: Optional[str] = Query(None, description="Filter by model name"),
    run_name: Optional[str] = Query(None, description="Filter by run/service name"),
    project_name: Optional[str] = Query(None),
    period: str = Query("hour", description="Aggregation period: minute, hour, day, week, month, year"),
    start: Optional[str] = Query(None, description="ISO start datetime"),
    end: Optional[str] = Query(None, description="ISO end datetime"),
):
    """Return time-bucketed metrics for a model/service.

    Each bucket contains:
    - tokens_per_sec (completion_tokens / latency aggregated)
    - avg_latency_ms
    - total_tokens
    - request_count
    - success_rate (fraction of 2xx responses)
    """
    if period not in _PERIODS:
        return {"error": f"Invalid period. Choose from: {', '.join(sorted(_PERIODS))}"}

    now = datetime.now(timezone.utc)
    t_end = datetime.fromisoformat(end) if end else now
    t_start = datetime.fromisoformat(start) if start else (t_end - _DEFAULT_RANGE[period])

    bucket = func.date_trunc(period, ModelMetric.created_at).label("bucket")

    success_count = func.count(
        case((ModelMetric.status_code < 400, ModelMetric.id))
    )
    total_count = func.count(ModelMetric.id)

    stmt = (
        select(
            bucket,
            func.sum(ModelMetric.completion_tokens).label("total_completion_tokens"),
            func.sum(ModelMetric.prompt_tokens).label("total_prompt_tokens"),
            func.sum(ModelMetric.total_tokens).label("total_tokens"),
            func.avg(ModelMetric.latency_ms).label("avg_latency_ms"),
            total_count.label("request_count"),
            success_count.label("success_count"),
            # tokens/sec: total completion tokens / total latency seconds
            case(
                (func.sum(ModelMetric.latency_ms) > 0,
                 cast(func.sum(ModelMetric.completion_tokens), Float) /
                 (func.sum(ModelMetric.latency_ms) / 1000.0)),
                else_=0.0,
            ).label("tokens_per_sec"),
        )
        .where(ModelMetric.created_at >= t_start)
        .where(ModelMetric.created_at <= t_end)
        .group_by(bucket)
        .order_by(bucket)
    )

    if model:
        stmt = stmt.where(ModelMetric.model == model)
    if run_name:
        stmt = stmt.where(ModelMetric.run_name == run_name)
    if project_name:
        stmt = stmt.where(ModelMetric.project_name == project_name)

    rows = (await session.execute(stmt)).all()

    buckets = []
    for r in rows:
        req_count = r.request_count or 0
        ok_count = r.success_count or 0
        buckets.append({
            "timestamp": r.bucket.isoformat() if r.bucket else None,
            "tokens_per_sec": round(r.tokens_per_sec or 0, 2),
            "avg_latency_ms": round(r.avg_latency_ms or 0, 1),
            "total_tokens": r.total_tokens or 0,
            "total_prompt_tokens": r.total_prompt_tokens or 0,
            "total_completion_tokens": r.total_completion_tokens or 0,
            "request_count": req_count,
            "success_rate": round(ok_count / req_count, 4) if req_count > 0 else 1.0,
        })

    return {
        "model": model,
        "run_name": run_name,
        "period": period,
        "start": t_start.isoformat(),
        "end": t_end.isoformat(),
        "buckets": buckets,
    }


@router.get("/summary")
async def get_metrics_summary(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    model: Optional[str] = Query(None),
    run_name: Optional[str] = Query(None),
    project_name: Optional[str] = Query(None),
    hours: int = Query(24, description="Lookback window in hours"),
):
    """Return a single-row summary of metrics over the lookback window."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    success_count = func.count(
        case((ModelMetric.status_code < 400, ModelMetric.id))
    )
    total_count = func.count(ModelMetric.id)

    stmt = (
        select(
            func.sum(ModelMetric.total_tokens).label("total_tokens"),
            func.sum(ModelMetric.prompt_tokens).label("total_prompt_tokens"),
            func.sum(ModelMetric.completion_tokens).label("total_completion_tokens"),
            func.avg(ModelMetric.latency_ms).label("avg_latency_ms"),
            total_count.label("request_count"),
            success_count.label("success_count"),
            case(
                (func.sum(ModelMetric.latency_ms) > 0,
                 cast(func.sum(ModelMetric.completion_tokens), Float) /
                 (func.sum(ModelMetric.latency_ms) / 1000.0)),
                else_=0.0,
            ).label("tokens_per_sec"),
        )
        .where(ModelMetric.created_at >= since)
    )

    if model:
        stmt = stmt.where(ModelMetric.model == model)
    if run_name:
        stmt = stmt.where(ModelMetric.run_name == run_name)
    if project_name:
        stmt = stmt.where(ModelMetric.project_name == project_name)

    r = (await session.execute(stmt)).one()
    req_count = r.request_count or 0
    ok_count = r.success_count or 0

    return {
        "model": model,
        "run_name": run_name,
        "hours": hours,
        "tokens_per_sec": round(r.tokens_per_sec or 0, 2),
        "avg_latency_ms": round(r.avg_latency_ms or 0, 1),
        "total_tokens": r.total_tokens or 0,
        "total_prompt_tokens": r.total_prompt_tokens or 0,
        "total_completion_tokens": r.total_completion_tokens or 0,
        "request_count": req_count,
        "success_rate": round(ok_count / req_count, 4) if req_count > 0 else 1.0,
    }
