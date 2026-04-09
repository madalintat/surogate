// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";

// ── Types ──────────────────────────────────────────────────────────

export interface AgentResponse {
  id: string;
  project_id: string;
  project_name: string;
  name: string;
  harness: string;
  display_name: string;
  description: string;
  version: string;
  model_id: string;
  model_name: string;
  status: string;
  replicas: Record<string, number> | null;
  image: string;
  endpoint: string;
  system_prompt: string;
  env_vars: Record<string, string> | null;
  resources: Record<string, string> | null;
  created_by_id: string;
  created_by_username: string;
  hub_ref: string | null;
  created_at: string | null;
}

export interface AgentListResponse {
  agents: AgentResponse[];
  total: number;
}

export interface AgentCreateRequest {
  name: string;
  harness: string;
  display_name: string;
  description?: string;
  version?: string;
  model_id: string;
  status?: string;
  replicas?: Record<string, number>;
  image?: string;
  endpoint?: string;
  system_prompt?: string;
  env_vars?: Record<string, string>;
  resources?: Record<string, string>;
}

export interface AgentUpdateRequest {
  name?: string;
  harness?: string;
  display_name?: string;
  description?: string;
  version?: string;
  model_id?: string;
  status?: string;
  replicas?: Record<string, number>;
  image?: string;
  endpoint?: string;
  system_prompt?: string;
  env_vars?: Record<string, string>;
  resources?: Record<string, string>;
}

// ── API calls ──────────────────────────────────────────────────────

export async function listAgents(
  projectId?: string,
  status?: string,
  harness?: string,
  limit?: number,
): Promise<AgentListResponse> {
  const params = new URLSearchParams();
  if (projectId) params.append("project_id", projectId);
  if (status) params.append("status", status);
  if (harness) params.append("harness", harness);
  if (limit != null) params.append("limit", String(limit));
  const response = await authFetch(`/api/agents/agents?${params}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to fetch agents");
  }
  return (await response.json()) as AgentListResponse;
}

export async function getAgent(agentId: string): Promise<AgentResponse> {
  const response = await authFetch(`/api/agents/agents/${agentId}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to fetch agent");
  }
  return (await response.json()) as AgentResponse;
}

export async function createAgent(
  projectId: string,
  body: AgentCreateRequest,
): Promise<AgentResponse> {
  const params = new URLSearchParams({ project_id: projectId });
  const response = await authFetch(`/api/agents/agents?${params}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to create agent");
  }
  return (await response.json()) as AgentResponse;
}

export async function updateAgent(
  agentId: string,
  body: AgentUpdateRequest,
): Promise<AgentResponse> {
  const response = await authFetch(`/api/agents/agents/${agentId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to update agent");
  }
  return (await response.json()) as AgentResponse;
}

export async function deleteAgent(agentId: string): Promise<void> {
  const response = await authFetch(`/api/agents/agents/${agentId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to delete agent");
  }
}
