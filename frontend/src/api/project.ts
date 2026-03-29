// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";
import type { Project } from "@/types/platform";

export async function fetchProjects(): Promise<Project[]> {
  const response = await authFetch("/api/projects");

  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to fetch projects");
  }

  return (await response.json()) as Project[];
}
