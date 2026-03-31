"""Repository functions for the agent/operate domain."""

from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from surogate.core.db.models.operate import Skill, SkillStatus


# ── Skill ────────────────────────────────────────────────────────────


async def create_skill(
    session: AsyncSession,
    *,
    name: str,
    display_name: str,
    project_id: str,
    author_id: str,
    description: str = "",
    content: str = "",
    version: str = "1.0.0",
    status: SkillStatus = SkillStatus.active,
    tags: list[str] | None = None,
    hub_ref: str | None = None,
) -> Skill:
    skill = Skill(
        name=name,
        display_name=display_name,
        description=description,
        content=content,
        version=version,
        status=status,
        project_id=project_id,
        author_id=author_id,
        tags=tags or [],
        hub_ref=hub_ref,
    )
    session.add(skill)
    await session.commit()
    return skill


async def get_skill(
    session: AsyncSession, skill_id: str
) -> Optional[Skill]:
    result = await session.execute(
        sa.select(Skill).where(Skill.id == skill_id)
    )
    return result.scalar_one_or_none()


async def list_skills(
    session: AsyncSession,
    *,
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> list[Skill]:
    stmt = sa.select(Skill).order_by(Skill.created_at.desc())
    if project_id is not None:
        stmt = stmt.where(Skill.project_id == project_id)
    if status is not None:
        stmt = stmt.where(Skill.status == status)
    stmt = stmt.limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_skill(
    session: AsyncSession,
    skill_id: str,
    **fields: object,
) -> Optional[Skill]:
    # Map string status to enum if provided
    if "status" in fields and isinstance(fields["status"], str):
        fields["status"] = SkillStatus(fields["status"])

    await session.execute(
        sa.update(Skill).where(Skill.id == skill_id).values(**fields)
    )
    await session.commit()
    result = await session.execute(
        sa.select(Skill).where(Skill.id == skill_id)
    )
    return result.scalar_one_or_none()


async def delete_skill(
    session: AsyncSession, skill_id: str
) -> bool:
    result = await session.execute(
        sa.delete(Skill).where(Skill.id == skill_id)
    )
    await session.commit()
    return result.rowcount > 0
