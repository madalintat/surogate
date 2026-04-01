from fastapi import APIRouter, Request
from prometheus_api_client import PrometheusConnect, MetricsList

from surogate.server.models.metrics import NodeMetricsResponse

router = APIRouter()

@router.get("/nodes", response_model=list[NodeMetricsResponse])
async def get_local_node_metrics(
    request: Request,
):
    prom = PrometheusConnect(url=request.app.state.config.prometheus_endpoint, disable_ssl=True)
    
    node_mem = prom.custom_query(query="node_memory_MemAvailable_bytes * on(pod) group_left(node) kube_pod_info")
    
    node_metrics = []
    for metric in MetricsList(node_mem):
        node_metrics.append(NodeMetricsResponse(
            timestamp=int(metric.end_time.timestamp()),
            node_name=metric.label_config['node'],
            free_memory_bytes=int(metric.metric_values.iloc[-1, 1])
        ))
        
    return node_metrics