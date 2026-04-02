// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { LocalTask } from "@/api/tasks";
import * as tasksApi from "@/api/tasks";
import type { AppState } from "./app-store";

export type TasksSlice = {
  tasks: LocalTask[];
  addTask: (task: LocalTask) => void;
  fetchTasks: () => Promise<void>;
  cancelTask: (taskId: string) => Promise<void>;
  deleteTask: (taskId: string) => Promise<void>;
};

export const createTasksSlice: StateCreator<AppState, [], [], TasksSlice> = (set, get) => ({
  tasks: [],

  addTask: (task) =>
    set((s) => {
      const tasks = [task, ...s.tasks.filter((t) => t.id !== task.id)];
      return { tasks };
    }),

  fetchTasks: async () => {
    try {
      const tasks = await tasksApi.listTasks();
      set({ tasks });
    } catch {
      // silently ignore fetch errors
    }
  },

  cancelTask: async (taskId) => {
    await tasksApi.cancelTask(taskId);
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === taskId ? { ...t, status: "cancelled" } : t,
      ),
    }));
  },

  deleteTask: async (taskId) => {
    await tasksApi.deleteTask(taskId);
    set((s) => ({ tasks: s.tasks.filter((t) => t.id !== taskId) }));
  },
});
