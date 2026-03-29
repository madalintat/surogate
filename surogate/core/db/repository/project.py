import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.platform import (
    Project,
    ProjectMember,
    ProjectMemberRole,
    ProjectStatus,
    User,
)


async def get_user_projects(
    session: AsyncSession, username: str
) -> list[Project]:
    """Return a list of projects the user belongs to (either as creator or member)."""
    result = await session.execute(
        sa.select(Project)
        .join(ProjectMember, ProjectMember.project_id == Project.id)
        .join(User, User.id == ProjectMember.user_id)
        .where(User.username == username)
    )
    return list(result.scalars().all())


async def seed_default_project(session: AsyncSession, username: str) -> None:
    """Create a default project for a user if they don't have any."""
    user = await session.execute(
        sa.select(User).where(User.username == username)
    )
    user_row = user.scalar_one_or_none()
    if user_row is None:
        return

    count = await session.execute(
        sa.select(sa.func.count())
        .select_from(ProjectMember)
        .where(ProjectMember.user_id == user_row.id)
    )
    if count.scalar_one() > 0:
        return

    project = Project(
        name="My First Project",
        namespace="default",
        color="#F59E0B",
        status=ProjectStatus.active,
        created_by_id=user_row.id,
    )
    session.add(project)
    await session.flush()

    member = ProjectMember(
        project_id=project.id,
        user_id=user_row.id,
        role=ProjectMemberRole.owner,
        added_by_id=user_row.id,
    )
    session.add(member)
    await session.commit()
