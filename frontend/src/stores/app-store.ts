// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { create } from "zustand";
import type { Project } from "@/types/platform";

type AppState = {
  /* projects */
  projects: Project[];
  activeProjectId: string | null;

  setProjects: (projects: Project[]) => void;
  setActiveProject: (id: string) => void;
  addProject: (project: Project) => void;
};

const SAMPLE_PROJECTS: Project[] = [
  { id: "cx-agent", name: "CX Support Agent", namespace: "prod-cx", color: "#F59E0B", status: "active", created_by_id: "", created_at: "" },
  { id: "code-assist", name: "Code Assistant", namespace: "prod-code", color: "#3B82F6", status: "active", created_by_id: "", created_at: "" },
  { id: "data-analyst", name: "Data Analyst Agent", namespace: "staging-da", color: "#8B5CF6", status: "active", created_by_id: "", created_at: "" },
];

export const useAppStore = create<AppState>((set) => ({
  projects: SAMPLE_PROJECTS,
  activeProjectId: SAMPLE_PROJECTS[0].id,

  setProjects: (projects) => set({ projects }),
  setActiveProject: (id) => set({ activeProjectId: id }),
  addProject: (project) =>
    set((s) => ({ projects: [...s.projects, project] })),
}));
