"""Compute module — SkyPilot initialization and monkeypatches.

SkyPilot is used as a direct library (bypassing its REST API server).
We call the implementation layer: sky.execution, sky.core, sky.jobs.server.core.

Monkeypatches are applied before any sky imports to support
platform-specific features.
"""

import os
import uuid

from surogate.utils.logger import get_logger

logger = get_logger()

_initialized = False

dummy_request_id = str(uuid.uuid4())  # Used in skylet for non-SkyPilot requests

def init_skypilot():
    """Initialize SkyPilot as a library.

    1. Redirect ~/.sky → ~/.surogate/sky (env var + constant patches).
    2. Initialize global user state (SkyPilot's SQLite DB).
    3. Reload SkyPilot config.
    """
    
    global _initialized
    if _initialized:
        return

    os.environ.setdefault("ENV_VAR_IS_SKYPILOT_SERVER", "1")
    
    # ── Patch get_current_request_id to return a real UUID ──────
    from sky.utils import common_utils
    _original_get_request_id = common_utils.get_current_request_id

    def _patched_get_current_request_id() -> str:
        value = _original_get_request_id()
        if value == "dummy-request-id":
            return dummy_request_id
        return value

    common_utils.get_current_request_id = _patched_get_current_request_id

    # ── SkyPilot state initialization ────────────────────────────
    from sky import global_user_state
    from sky import skypilot_config

    global_user_state.initialize_and_get_db()
    skypilot_config.safe_reload_config()

    _initialized = True
    logger.info("SkyPilot initialized (direct library mode)")

    from surogate.core.compute.kubernetes import init_kubernetes
    init_kubernetes()
    
