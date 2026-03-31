// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { Repository, RepositoryCreation, Ref, Commit, ObjectStats } from "@/types/hub";
import * as hubApi from "@/api/hub";
import type { AppState } from "./app-store";

export type HubSlice = {
  repositories: Repository[];
  currentRepo: Repository | null;
  branches: Record<string, Ref[]>;
  tags: Record<string, Ref[]>;
  commits: Record<string, Commit[]>;
  objects: Record<string, ObjectStats[]>;

  createRepository: (req: RepositoryCreation) => Promise<boolean>;
  deleteRepository: (repository: string) => Promise<boolean>;
  fetchRepositories: (prefix?: string) => Promise<void>;
  fetchRepository: (repository: string) => Promise<void>;
  fetchBranches: (repository: string) => Promise<void>;
  fetchTags: (repository: string) => Promise<void>;
  fetchCommits: (repository: string, ref: string) => Promise<void>;
  fetchObjects: (repository: string, ref: string, prefix?: string) => Promise<void>;
};

export const createHubSlice: StateCreator<AppState, [], [], HubSlice> = (set) => ({
  repositories: [],
  currentRepo: null,
  branches: {},
  tags: {},
  commits: {},
  objects: {},

  createRepository: async (req: RepositoryCreation) => {
    set({ error: null });
    try {
      const repo = await hubApi.createRepository(req);
      set((s) => ({ repositories: [...s.repositories, repo] }));
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  deleteRepository: async (repository: string) => {
    set({ error: null });
    try {
      await hubApi.deleteRepository(repository);
      set((s) => ({
        repositories: s.repositories.filter((r) => r.id !== repository),
        currentRepo: null,
      }));
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  fetchRepositories: async (prefix?: string) => {
    set({ loading: true, error: null });
    try {
      const res = await hubApi.listRepositories(prefix);
      set({ repositories: res.results, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  fetchRepository: async (repository: string) => {
    set((s) => ({
      loading: true,
      error: null,
      // Keep current repo visible while refreshing if it's the same one
      currentRepo: s.currentRepo?.id === repository ? s.currentRepo : null,
    }));
    try {
      const repo = await hubApi.getRepository(repository);
      set({ currentRepo: repo, loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  fetchBranches: async (repository: string) => {
    try {
      const res = await hubApi.listBranches(repository);
      set((s) => ({ branches: { ...s.branches, [repository]: res.results } }));
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchTags: async (repository: string) => {
    try {
      const res = await hubApi.listTags(repository);
      set((s) => ({ tags: { ...s.tags, [repository]: res.results } }));
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchCommits: async (repository: string, ref: string) => {
    const key = `${repository}:${ref}`;
    try {
      const res = await hubApi.listCommits(repository, ref);
      set((s) => ({ commits: { ...s.commits, [key]: res.results } }));
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchObjects: async (repository: string, ref: string, prefix?: string) => {
    const key = `${repository}:${ref}`;
    try {
      const res = await hubApi.listObjects(repository, ref, prefix);
      set((s) => ({ objects: { ...s.objects, [key]: res.results } }));
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },
});
