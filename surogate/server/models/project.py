"""Pydantic models for project API routes."""

from datetime import datetime

from pydantic import BaseModel


class ProjectResponse(BaseModel):
    id: str
    name: str
    namespace: str
    color: str
    status: str
    created_by_id: str
    created_at: datetime

    model_config = {"from_attributes": True}
