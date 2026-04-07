import type { K8Node } from "@/types/compute";
import { authFetch } from "@/api/auth";

export async function fetchK8Nodes(): Promise<K8Node[]> {
    const response = await authFetch(`/api/compute/nodes`);
    return (await response.json()) as K8Node[];
}

export type CloudBackend = {
    type: string;
    id: string;
    active_instances: number;
    hourly_cost: number;
};

export async function fetchCloudBackends(projectId: string): Promise<CloudBackend[]> {
    const response = await authFetch(`/api/compute/cloud/backends?project_id=${encodeURIComponent(projectId)}`);
    if (!response.ok) return [];
    return (await response.json()) as CloudBackend[];
}

export async function deleteCloudBackend(projectId: string, backendType: string): Promise<void> {
    const response = await authFetch(
        `/api/compute/cloud/backends/${encodeURIComponent(backendType)}?project_id=${encodeURIComponent(projectId)}`,
        { method: "DELETE" },
    );
    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || "Failed to delete backend");
    }
}

export type InstanceOffer = {
    backend: string;
    instance: string;
    region: string;
    price: number;
    cpus: number;
    memory_mib: number;
    spot: boolean;
    gpu_count: number;
    gpu_name: string | null;
    gpu_memory_mib: number | null;
    availability: string;
};

export async function fetchBackendOffers(projectId: string): Promise<InstanceOffer[]> {
    const response = await authFetch(`/api/compute/cloud/backends/offers?project_id=${encodeURIComponent(projectId)}`);
    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || "Failed to fetch offers");
    }
    return (await response.json()) as InstanceOffer[];
}

export type CloudInstance = {
    id: string;
    provider: string;
    region: string;
    instance_type: string;
    gpu: string;
    status: string;
    workload: string;
    started_at: string | null;
    cost_per_hour: number;
    estimated_total: number;
    spot_instance: boolean;
    project_id: string;
    project_name: string;
};

export async function fetchCloudInstances(): Promise<CloudInstance[]> {
    const response = await authFetch(`/api/compute/cloud/instances`);
    if (!response.ok) return [];
    return (await response.json()) as CloudInstance[];
}

export async function terminateCloudInstance(instanceId: string, projectName: string): Promise<void> {
    await authFetch(
        `/api/compute/cloud/instances/${encodeURIComponent(instanceId)}/terminate?project_name=${encodeURIComponent(projectName)}`,
        { method: "POST" },
    );
}

export type ConnectBackendResult = {
    status: string;
    provider: string;
    offers: InstanceOffer[];
};

export async function connectNebiusBackend(
    projectId: string,
    serviceAccountId: string,
    publicKeyId: string,
    privateKeyContent: string,
): Promise<ConnectBackendResult> {
    const response = await authFetch(`/api/compute/cloud/backends/nebius`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            project_id: projectId,
            service_account_id: serviceAccountId,
            public_key_id: publicKeyId,
            private_key_content: privateKeyContent,
        }),
    });
    if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(err.detail || "Failed to connect Nebius backend");
    }
    return (await response.json()) as ConnectBackendResult;
}