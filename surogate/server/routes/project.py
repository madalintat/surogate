"""
Project API routes
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.engine import get_session
from surogate.core.db.repository import project as project_repository
from surogate.server.auth.authentication import get_current_subject
from surogate.server.models.project import ProjectResponse

router = APIRouter()

@router.get("/projects", response_model=list[ProjectResponse])
async def get_user_projects(
    current_subject: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> list[ProjectResponse]:
    """Fetch the list of projects the user has access to."""
    projects = await project_repository.get_user_projects(session, current_subject)
    return [ProjectResponse.model_validate(p) for p in projects]
