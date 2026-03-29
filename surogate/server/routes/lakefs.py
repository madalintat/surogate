"""
Data Hub API routes
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from lakefs_sdk import Repository, RepositoryList, RefList, Ref, CommitList, Commit, ObjectStatsList, ObjectStats
from sqlalchemy.ext.asyncio import AsyncSession

import surogate.core.hub.lakefs as lakefs
from surogate.core.db.engine import get_session

from surogate.server.auth.authentication import get_current_subject

router = APIRouter()

# ============ Repositories ============

@router.get("/repositories", response_model=list[RepositoryList])
async def list_repos(
    request: Request,
    prefix: Optional[str] = Query(None),
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> list[RepositoryList]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.list_repositories(api_client, prefix)

@router.post("/repositories")
async def create_repo(
    request: Request,
    repo_name: str = Query(...),
    repo_type: str = Query(..., regex="^(model|dataset)$"),
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[Repository]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.create_repository(api_client, repo_name, repo_type)

@router.get("/repositories/{repository}")
async def get_repo(
    repository: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[Repository]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_repository(api_client, repository)

@router.delete("/repositories/{repository}")
async def delete_repo(
    repository: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> dict:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    result = await lakefs.delete_repository(api_client, repository)
    return {"success": result}

# ============ Branches ============
@router.get("/repositories/{repository}/branches")
async def list_branches(
    repository: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> RefList:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_branches(api_client, repository)

@router.post("/repositories/{repository}/branches")
async def create_branch(
    repository: str,
    request: Request,
    branch: str= Query(...),
    source: Optional[str] = Query(None),
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[str]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.create_branch(api_client, repository, branch, source=source)

@router.get("/repositories/{repository}/branches/{branch}")
async def get_branch(
    repository: str,
    branch: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[Ref]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_branch(api_client, repository, branch)

@router.delete("/repositories/{repository}/branches/{branch}")
async def delete_branch(
    repository: str,
    branch: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> dict:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    result = await lakefs.delete_branch(api_client, repository, branch)
    return {"success": result}

# ============ Tags ============
@router.get("/repositories/{repository}/tags")
async def list_tags(
    repository: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> RefList:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_tags(api_client, repository)

@router.post("/repositories/{repository}/tags")
async def create_tag(
    repository: str,
    request: Request,
    tag: str= Query(...),
    commit: Optional[str] = Query(None),
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[str]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.create_tag(api_client, repository, tag, commit=commit)

@router.get("/repositories/{repository}/tags/{tag}")
async def get_tag(
    repository: str,
    tag: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[Ref]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_tag(api_client, repository, tag)

@router.delete("/repositories/{repository}/tags/{tag}")
async def delete_tag(
    repository: str,
    tag: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> dict:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    result = await lakefs.delete_tag(api_client, repository, tag)
    return {"success": result}

# ============ Commits ============
@router.get("/repositories/{repository}/refs/{ref}/commits")
async def list_commits(
    repository: str,
    ref: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> CommitList:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_commits(api_client, repository, ref)

@router.get("/repositories/{repository}/commits/{commitId}")
async def get_commit(
    repository: str,
    commitId: str,
    request: Request,
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[Commit]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_commit(api_client, repository, commitId)

# ============ Objects ============
@router.get("/repositories/{repository}/refs/{ref}/objects/ls")
async def list_objects(
    repository: str,
    ref: str,
    request: Request,
    prefix: Optional[str] = Query(None),
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> ObjectStatsList:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_objects(api_client, repository, ref, prefix=prefix)

@router.get("/repositories/{repository}/refs/{ref}/objects")
async def get_object(
    repository: str,
    ref: str,
    request: Request,
    path: Optional[str] = Query(...),
    user: str = Depends(get_current_subject),
    session: AsyncSession = Depends(get_session),
) -> Optional[ObjectStats]:
    api_client = await lakefs.get_lakefs_client(user, session, request.app.state.config)
    return await lakefs.get_object(api_client, repository, ref, path)