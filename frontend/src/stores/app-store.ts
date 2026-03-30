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

export const useAppStore = create<AppState>((set) => ({
  projects: [],
  activeProjectId: null,

  setProjects: (projects) => set({ projects }),
  setActiveProject: (id) => set({ activeProjectId: id }),
  addProject: (project) =>
    set((s) => ({ projects: [...s.projects, project] })),
}));
