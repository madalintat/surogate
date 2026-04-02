from sky.models import KubernetesNodesInfo

k8_nodes: KubernetesNodesInfo = None
context: None

def init_kubernetes():
    # these imports must be done inside the function to avoid importing sky before skypilot is initialized,
    # which would break context propagation for threads.
    from sky.provision.kubernetes.utils import get_current_kube_config_context_name, get_kubernetes_node_info

    global k8_nodes, context
    context = get_current_kube_config_context_name()
    k8_nodes = get_kubernetes_node_info(context=context)
    
    if k8_nodes is None:
        raise RuntimeError("Failed to load Kubernetes node info on startup.")