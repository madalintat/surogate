"""Repository functions for the Agent domain."""

from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from surogate.core.db.models.operate import Agent, AgentStatus, AgentVersion, AgentVersionStatus


# ── Agent CRUD ──────────────────────────────────────────────────────


async def create_agent(
    session: AsyncSession,
    *,
    project_id: str,
    name: str,
    harness: str,
    display_name: str,
    description: str = "",
    version: str = "0.1.0",
    model_id: str | None = None,
    status: AgentStatus = AgentStatus.stopped,
    replicas: dict | None = None,
    image: str = "",
    endpoint: str = "",
    system_prompt: str = "",
    env_vars: dict | None = None,
    resources: dict | None = None,
    created_by_id: str,
    hub_ref: str | None = None,
) -> Agent:
    agent = Agent(
        project_id=project_id,
        name=name,
        harness=harness,
        display_name=display_name,
        description=description,
        version=version,
        model_id=model_id,
        status=status,
        replicas=replicas,
        image=image,
        endpoint=endpoint,
        system_prompt=system_prompt,
        env_vars=env_vars,
        resources=resources,
        created_by_id=created_by_id,
        hub_ref=hub_ref,
    )
    session.add(agent)
    await session.commit()
    return await get_agent(session, agent.id)


async def get_agent(
    session: AsyncSession, agent_id: str
) -> Optional[Agent]:
    result = await session.execute(
        sa.select(Agent)
        .where(Agent.id == agent_id)
        .options(
            selectinload(Agent.project),
            selectinload(Agent.model),
            selectinload(Agent.created_by),
            selectinload(Agent.versions),
            selectinload(Agent.skills),
            selectinload(Agent.tools),
            selectinload(Agent.mcp_servers),
        )
    )
    return result.scalar_one_or_none()


async def list_agents(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    harness: Optional[str] = None,
    limit: int = 100,
) -> list[Agent]:
    stmt = (
        sa.select(Agent)
        .options(
            selectinload(Agent.project),
            selectinload(Agent.model),
            selectinload(Agent.created_by),
        )
        .order_by(Agent.created_at.desc())
    )
    if project_id is not None:
        stmt = stmt.where(Agent.project_id == project_id)
    if status is not None:
        stmt = stmt.where(Agent.status == AgentStatus(status))
    if harness is not None:
        stmt = stmt.where(Agent.harness == harness)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_agent(
    session: AsyncSession,
    agent_id: str,
    **fields: object,
) -> Optional[Agent]:
    if "status" in fields and isinstance(fields["status"], str):
        fields["status"] = AgentStatus(fields["status"])

    await session.execute(
        sa.update(Agent).where(Agent.id == agent_id).values(**fields)
    )
    await session.commit()
    return await get_agent(session, agent_id)


async def delete_agent(
    session: AsyncSession, agent_id: str
) -> bool:
    # Delete associated versions first
    await session.execute(
        sa.delete(AgentVersion).where(AgentVersion.agent_id == agent_id)
    )
    result = await session.execute(
        sa.delete(Agent).where(Agent.id == agent_id)
    )
    await session.commit()
    return result.rowcount > 0
