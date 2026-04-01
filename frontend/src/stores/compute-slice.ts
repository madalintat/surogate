import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";
import type { K8Node } from "@/types/compute";
import * as computeApi from "@/api/compute";

let pollTimer: ReturnType<typeof setInterval> | null = null;
let pollRefCount = 0;

export type ComputeSlice = {
    k8sNodes: K8Node[];

    fetchK8Nodes: () => Promise<void>;
    startK8sPolling: () => () => void;
}

export const createComputeSlice: StateCreator<AppState, [], [], ComputeSlice> = (set, get) => ({
    k8sNodes: [] as K8Node[],

    fetchK8Nodes: async () => {
        try {
            const res = await computeApi.fetchK8Nodes();
            set({ k8sNodes: res });
        } catch (e) {
            set({ error: (e as Error).message });
        }
    },

    startK8sPolling: () => {
        pollRefCount++;
        if (!pollTimer) {
            get().fetchK8Nodes();
            pollTimer = setInterval(() => get().fetchK8Nodes(), 5000);
        }
        return () => {
            pollRefCount--;
            if (pollRefCount <= 0 && pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
                pollRefCount = 0;
            }
        };
    },
});