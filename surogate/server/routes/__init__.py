"""
API Routes
"""
from routes.auth import router as auth_router
from routes.project import router as project_router

__all__ = [
    "auth_router",
    "project_router",
]
