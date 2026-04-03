from contextvars import ContextVar
from typing import List

from sky.clouds import cloud as sky_cloud
from sky import clouds as sky_clouds

# Thread-local variables set by surogate.core.compute.sky.launch_serving_service
_active_project_id: ContextVar[str] = ContextVar("_active_project_id")
_allowed_clouds: ContextVar[List[str]] = ContextVar("_allowed_clouds")
_always_allowed_clouds = ['Kubernetes']

def _surogate_get_workspace_allowed_clouds(workspace: str) -> List[str]:
    project_allowed_clouds = _allowed_clouds.get([])
    return _always_allowed_clouds + project_allowed_clouds

def _surogate_get_allowed_clouds(workspace: str) -> List[str]:
    return _surogate_get_workspace_allowed_clouds(workspace)

def _surogate_get_active_workspace(force_user_workspace: bool = False) -> str:
    from sky.skylet import constants
    return _active_project_id.get("") or constants.SKYPILOT_DEFAULT_WORKSPACE

def _surogate_get_cached_enabled_storage_cloud_names_or_refresh(
    raise_if_no_cloud_access: bool = False
) -> List[str]:
    from sky.exceptions import NoCloudAccessError
    if raise_if_no_cloud_access:
        raise NoCloudAccessError('No cloud access available for storage.')
    return []

def _surogate_get_cached_enabled_clouds(
    cloud_capability: 'sky_cloud.CloudCapability',
    workspace: str
) -> List['sky_clouds.Cloud']:
    from sky.utils import registry
    project_allowed_clouds = _allowed_clouds.get([])
    clouds = [registry.CLOUD_REGISTRY.from_str(c) for c in _always_allowed_clouds + project_allowed_clouds]
    return clouds


