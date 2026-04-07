"""Record and query model inference metrics."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.metrics import ModelMetric


async def record_metric(
    session: AsyncSession,
    *,
    project_name: str,
    run_name: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    latency_ms: float = 0.0,
    status_code: int = 200,
    is_streaming: bool = False,
) -> ModelMetric:
    row = ModelMetric(
        project_name=project_name,
        run_name=run_name,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        status_code=status_code,
        is_streaming=is_streaming,
    )
    session.add(row)
    await session.commit()
    return row
