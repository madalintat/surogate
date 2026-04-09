"""Agent CRUD API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.engine import get_session
from surogate.core.db.models.operate import AgentStatus
from surogate.core.db.repository import agents as agent_repo
from surogate.core.db.repository import user as user_repo
from surogate.server.auth.authentication import get_current_subject
from surogate.server.models.agent import (
    AgentCreateRequest,
    AgentListResponse,
    AgentResponse,
    AgentUpdateRequest,
)

router = APIRouter()

# ── helpers ──────────────────────────────────────────────────────────


def _agent_to_response(agent) -> AgentResponse:
    return AgentResponse(
        id=agent.id,
        project_id=agent.project_id,
        project_name=agent.project.name if agent.project else "",
        name=agent.name,
        harness=agent.harness,
        display_name=agent.display_name,
        description=agent.description or "",
        version=agent.version,
        model_id=agent.model_id,
        model_name=agent.model.name if agent.model else "",
        status=agent.status.value if hasattr(agent.status, "value") else agent.status,
        replicas=agent.replicas,
        image=agent.image or "",
        endpoint=agent.endpoint or "",
        system_prompt=agent.system_prompt or "",
        env_vars=agent.env_vars,
        resources=agent.resources,
        created_by_id=agent.created_by_id,
        created_by_username=agent.created_by.username if agent.created_by else "",
        hub_ref=agent.hub_ref,
        created_at=str(agent.created_at) if agent.created_at else None,
    )


# ── Agent CRUD ───────────────────────────────────────────────────────


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    harness: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
):
    agents = await agent_repo.list_agents(
        session,
        project_id=project_id,
        status=status,
        harness=harness,
        limit=limit,
    )
    return AgentListResponse(
        agents=[_agent_to_response(a) for a in agents],
        total=len(agents),
    )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    agent = await agent_repo.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_to_response(agent)


@router.post("/agents", response_model=AgentResponse, status_code=201)
async def create_agent(
    body: AgentCreateRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
    project_id: str = Query(...),
):
    user = await user_repo.get_user_by_username(session, current_subject)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    try:
        agent_status = AgentStatus(body.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")

    agent = await agent_repo.create_agent(
        session,
        project_id=project_id,
        name=body.name,
        harness=body.harness,
        display_name=body.display_name,
        description=body.description,
        version=body.version,
        model_id=body.model_id,
        status=agent_status,
        replicas=body.replicas,
        image=body.image,
        endpoint=body.endpoint,
        system_prompt=body.system_prompt,
        env_vars=body.env_vars,
        resources=body.resources,
        created_by_id=user.id,
    )
    return _agent_to_response(agent)


@router.patch("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    body: AgentUpdateRequest,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    existing = await agent_repo.get_agent(session, agent_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    fields = body.model_dump(exclude_unset=True)
    if not fields:
        return _agent_to_response(existing)

    agent = await agent_repo.update_agent(session, agent_id, **fields)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agent_to_response(agent)


@router.delete("/agents/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
):
    existing = await agent_repo.get_agent(session, agent_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    await agent_repo.delete_agent(session, agent_id)
