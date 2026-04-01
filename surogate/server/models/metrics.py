from pydantic import BaseModel

class NodeMetricsResponse(BaseModel):
    timestamp: int
    node_name: str
    free_memory_bytes: int
    