import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";
import type { K8Node } from "@/types/compute";
import type { CloudBackend, CloudInstance, InstanceOffer } from "@/api/compute";
import * as computeApi from "@/api/compute";

export type ComputeSlice = {
    k8sNodes: K8Node[];
    cloudBackends: CloudBackend[];
    cloudInstances: CloudInstance[];
    backendOffers: InstanceOffer[];
    backendOffersLoading: boolean;
    backendOffersError: string | null;

    fetchK8Nodes: () => Promise<void>;
    fetchCloudBackends: () => Promise<void>;
    fetchCloudInstances: () => Promise<void>;
    terminateCloudInstance: (instanceId: string, projectName: string) => Promise<void>;
    fetchBackendOffers: () => Promise<void>;
    setBackendOffers: (offers: InstanceOffer[]) => void;
    deleteCloudBackend: (backendType: string) => Promise<void>;
}

export const createComputeSlice: StateCreator<AppState, [], [], ComputeSlice> = (set, get) => ({
    k8sNodes: [] as K8Node[],
    cloudBackends: [] as CloudBackend[],
    cloudInstances: [] as CloudInstance[],
    backendOffers: [] as InstanceOffer[],
    backendOffersLoading: false,
    backendOffersError: null as string | null,

    fetchK8Nodes: async () => {
        try {
            const res = await computeApi.fetchK8Nodes();
            set({ k8sNodes: res });
        } catch (e) {
            set({ error: (e as Error).message });
        }
    },

    fetchCloudBackends: async () => {
        const projectId = get().activeProjectId;
        if (!projectId) return;
        try {
            const res = await computeApi.fetchCloudBackends(projectId);
            set({ cloudBackends: res });
        } catch (e) {
            set({ error: (e as Error).message });
        }
    },

    fetchCloudInstances: async () => {
        try {
            const res = await computeApi.fetchCloudInstances();
            set({ cloudInstances: res });
        } catch (e) {
            set({ error: (e as Error).message });
        }
    },

    terminateCloudInstance: async (instanceId, projectName) => {
        await computeApi.terminateCloudInstance(instanceId, projectName);
        set((s) => ({ cloudInstances: s.cloudInstances.filter((i) => i.id !== instanceId) }));
    },

    setBackendOffers: (offers) => set({ backendOffers: offers, backendOffersError: null }),

    deleteCloudBackend: async (backendType) => {
        const projectId = get().activeProjectId;
        if (!projectId) return;
        await computeApi.deleteCloudBackend(projectId, backendType);
        set((s) => ({ cloudBackends: s.cloudBackends.filter((b) => b.type !== backendType) }));
    },

    fetchBackendOffers: async () => {
        const projectId = get().activeProjectId;
        if (!projectId) return;
        set({ backendOffersLoading: true, backendOffersError: null });
        try {
            const res = await computeApi.fetchBackendOffers(projectId);
            set({ backendOffers: res });
        } catch (e) {
            set({ backendOffersError: (e as Error).message });
        } finally {
            set({ backendOffersLoading: false });
        }
    },
});
