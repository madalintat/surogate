from typing import Optional, List, Union

from sqlalchemy.ext.asyncio import AsyncSession
import lakefs_sdk
from lakefs_sdk import ApiClient, ApiException, RepositoryCreation, RepositoryList, Repository, RefList, Ref, BranchCreation, TagCreation, Commit, CommitList, ObjectStatsList, ObjectStats

from surogate.core.config.server_config import ServerConfig
from surogate.utils.logger import get_logger
import surogate.core.db.repository.user as user_repo

logger = get_logger()

REPO_TYPE_MODEL = "model"
REPO_TYPE_DATASET = "dataset"

async def get_lakefs_client(user: str, session: AsyncSession, config: ServerConfig) -> ApiClient:
    endpoint = config.lakefs_endpoint
    if "/api/v1" not in endpoint:
        endpoint = endpoint.rstrip("/") + "/api/v1"

    key, secret = await user_repo.get_lakefs_credentials(session, user)
    sdk_config = lakefs_sdk.Configuration(host=endpoint, username=key, password=secret)
    sdk_config.verify_ssl = False
    return ApiClient(sdk_config)

# ============ Repositories ============

async def list_repositories(client: ApiClient, prefix: Optional[str] = None) -> List[RepositoryList]:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)
        return repos_api.list_repositories(prefix=prefix)
    except ApiException as e:
        logger.error(f"Error listing LakeFS repositories: {e}")
        return []
    
async def create_repository(client: ApiClient, repository: str, type: str) -> Optional[Repository]:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)
        request = RepositoryCreation(
            name=repository,
            storage_namespace=f"local://{repository}",
            default_branch="main",
            metadata={"type": type},
        )
        return repos_api.create_repository(repository_creation=request)
    except ApiException as e:
        logger.error(f"Error creating LakeFS repository '{repository}': {e}")
        return None
    

async def get_repository(client: ApiClient, repository: str) -> Optional[Repository]:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)
        return repos_api.get_repository(repository=repository)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS repository '{repository}': {e}")
        return None
    

async def delete_repository(client: ApiClient, repository: str) -> bool:
    try:
        repos_api = lakefs_sdk.RepositoriesApi(client)
        repos_api.delete_repository(repository=repository, force=True)
        return True
    except ApiException as e:
        logger.error(f"Error deleting LakeFS repository '{repository}': {e}")
        return False
    

# ============ Branches ============
async def get_branches(client: ApiClient, repository: str) -> RefList:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        return branches_api.list_branches(repository=repository)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS branches for repository '{repository}': {e}")
        return RefList(pagination=None, results=[])

async def create_branch(client: ApiClient, repository: str, branch: str, source: Optional[str] = None) -> Optional[str]:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        request = BranchCreation(
            name=branch,
            source=source,
            force=False,
            hidden=False
        )
        return branches_api.create_branch(repository=repository, branch_creation=request)
    except ApiException as e:
        logger.error(f"Error creating LakeFS branch '{branch}' in repository '{repository}': {e}")
        return None
    
async def get_branch(client: ApiClient, repository: str, branch: str) -> Optional[Ref]:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        return branches_api.get_branch(repository=repository, branch=branch)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS branch '{branch}' in repository '{repository}': {e}")
        return None

async def delete_branch(client: ApiClient, repository: str, branch: str) -> bool:
    try:
        branches_api = lakefs_sdk.BranchesApi(client)
        branches_api.delete_branch(repository=repository, branch=branch, force=True)
        return True
    except ApiException as e:
        logger.error(f"Error deleting LakeFS branch '{branch}' in repository '{repository}': {e}")
        return False
    
# ============ Tags ============
async def get_tags(client: ApiClient, repository: str) -> RefList:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        return tags_api.list_tags(repository=repository)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS tags for repository '{repository}': {e}")
        return RefList(pagination=None, results=[])

async def create_tag(client: ApiClient, repository: str, tag: str, commit: Optional[str] = None) -> Optional[str]:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        request = TagCreation(
            id=tag,
            ref=commit,
        )
        return tags_api.create_tag(repository=repository, tag_creation=request)
    except ApiException as e:
        logger.error(f"Error creating LakeFS tag '{tag}' in repository '{repository}': {e}")
        return None
    
async def get_tag(client: ApiClient, repository: str, tag: str) -> Optional[Ref]:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        return tags_api.get_tag(repository=repository, tag=tag)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS tag '{tag}' in repository '{repository}': {e}")
        return None
    
async def delete_tag(client: ApiClient, repository: str, tag: str) -> bool:
    try:
        tags_api = lakefs_sdk.TagsApi(client)
        tags_api.delete_tag(repository=repository, tag=tag, force=True)
        return True
    except ApiException as e:
        logger.error(f"Error deleting LakeFS tag '{tag}' in repository '{repository}': {e}")
        return False
    
# ============ Commits ============
async def get_commits(client: ApiClient, repository: str, ref: str) -> CommitList:
    try:
        commits_api = lakefs_sdk.RefsApi(client)
        return commits_api.log_commits(repository=repository, ref=ref)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS commits for repository '{repository}' and ref '{ref}': {e}")
        return CommitList(pagination=None, results=[])

async def get_commit(client: ApiClient, repository: str, commit_id: str) -> Optional[Commit]:
    try:
        commits_api = lakefs_sdk.CommitsApi(client)
        return commits_api.get_commit(repository=repository, commit_id=commit_id)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS commit '{commit_id}' in repository '{repository}': {e}")
        return None
    
# ============ Objects ============
async def get_objects(client: ApiClient, repository: str, ref: str, prefix: Optional[str] = None) -> ObjectStatsList:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        return objects_api.list_objects(repository=repository, ref=ref, prefix=prefix).results
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS objects for repository '{repository}' and ref '{ref}': {e}")
        return []
    
async def delete_object(client: ApiClient, repository: str, branch: str, path: str) -> bool:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        objects_api.delete_object(repository=repository, ref=branch, path=path)
        return True
    except ApiException as e:
        logger.error(f"Error deleting LakeFS object '{path}' in repository '{repository}' and branch '{branch}': {e}")
        return False
    
async def get_object(client: ApiClient, repository: str, ref: str, path: str) -> Optional[ObjectStats]:
    try:
        objects_api = lakefs_sdk.ObjectsApi(client)
        return objects_api.get_object(repository=repository, ref=ref, path=path)
    except ApiException as e:
        logger.error(f"Error retrieving LakeFS object '{path}' in repository '{repository}' and ref '{ref}': {e}")
        return None
    