// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";
import type { Conversation, ConversationDetail, ConversationTurn, ConversationMessage } from "@/features/conversations/conversations-data";

// -- Raw backend types (snake_case) ----------------------------------------

interface RawConversationSummary {
  conversation_id: string;
  started_at: string | null;
  last_turn_at: string | null;
  turn_count: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  avg_latency_ms: number | null;
  model: string;
  project_name: string;
  run_name: string;
  has_compaction: boolean;
}

interface RawConversationListResponse {
  conversations: RawConversationSummary[];
  total: number;
}

interface RawTurn {
  id: string;
  parent_hash: string | null;
  state_hash: string;
  tail_hash: string;
  model: string;
  is_streaming: boolean;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  latency_ms: number | null;
  compacted: boolean;
  created_at: string | null;
}

interface RawMessage {
  role: "user" | "assistant";
  content: string;
  reasoning_content?: string | null;
  timestamp: string | null;
  tokens: number | null;
  latency: number | null;
}

interface RawConversationDetail {
  conversation_id: string;
  project_name: string;
  run_name: string;
  model: string;
  system_prompt: string | null;
  started_at: string | null;
  last_turn_at: string | null;
  turn_count: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  avg_latency_ms: number | null;
  has_compaction: boolean;
  messages: RawMessage[];
  turns: RawTurn[];
}

// -- Response types --------------------------------------------------------

export interface ConversationListResponse {
  conversations: Conversation[];
  total: number;
}

// -- Transform -------------------------------------------------------------

function formatRelativeTime(iso: string | null): string {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatDuration(startIso: string | null, endIso: string | null): string {
  if (!startIso || !endIso) return "";
  const ms = new Date(endIso).getTime() - new Date(startIso).getTime();
  const secs = Math.floor(ms / 1000);
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  const remSecs = secs % 60;
  return `${mins}m ${remSecs}s`;
}

function transformSummary(r: RawConversationSummary): Conversation {
  return {
    id: r.conversation_id,
    model: r.model,
    projectName: r.project_name,
    runName: r.run_name,
    turnCount: r.turn_count,
    tokensIn: r.total_prompt_tokens,
    tokensOut: r.total_completion_tokens,
    totalTokens: r.total_tokens,
    avgLatencyMs: r.avg_latency_ms,
    startedAt: r.started_at,
    lastTurnAt: r.last_turn_at,
    hasCompaction: r.has_compaction,
    time: formatRelativeTime(r.last_turn_at),
    duration: formatDuration(r.started_at, r.last_turn_at),
  };
}

function transformDetail(r: RawConversationDetail): ConversationDetail {
  return {
    id: r.conversation_id,
    model: r.model,
    projectName: r.project_name,
    runName: r.run_name,
    turnCount: r.turn_count,
    tokensIn: r.total_prompt_tokens,
    tokensOut: r.total_completion_tokens,
    totalTokens: r.total_tokens,
    avgLatencyMs: r.avg_latency_ms,
    startedAt: r.started_at,
    lastTurnAt: r.last_turn_at,
    hasCompaction: r.has_compaction,
    time: formatRelativeTime(r.last_turn_at),
    duration: formatDuration(r.started_at, r.last_turn_at),
    systemPrompt: r.system_prompt,
    messages: r.messages.map((m) => ({
      role: m.role,
      content: m.content,
      reasoningContent: m.reasoning_content ?? undefined,
      timestamp: m.timestamp,
      tokens: m.tokens ?? 0,
      latency: m.latency ?? undefined,
    })),
    turns: r.turns.map((t) => ({
      id: t.id,
      parentHash: t.parent_hash,
      stateHash: t.state_hash,
      tailHash: t.tail_hash,
      model: t.model,
      isStreaming: t.is_streaming,
      promptTokens: t.prompt_tokens,
      completionTokens: t.completion_tokens,
      totalTokens: t.total_tokens,
      latencyMs: t.latency_ms,
      compacted: t.compacted,
      createdAt: t.created_at,
    })),
  };
}

// -- API functions ---------------------------------------------------------

export async function listConversations(params?: {
  project_name?: string;
  model?: string;
  search?: string;
  limit?: number;
  offset?: number;
}): Promise<ConversationListResponse> {
  const qs = new URLSearchParams();
  if (params?.project_name) qs.set("project_name", params.project_name);
  if (params?.model) qs.set("model", params.model);
  if (params?.search) qs.set("search", params.search);
  if (params?.limit) qs.set("limit", String(params.limit));
  if (params?.offset) qs.set("offset", String(params.offset));

  const url = `/api/conversations/${qs.toString() ? `?${qs}` : ""}`;
  const response = await authFetch(url);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to list conversations");
  }
  const raw = (await response.json()) as RawConversationListResponse;
  return {
    conversations: raw.conversations.map(transformSummary),
    total: raw.total,
  };
}

export async function getConversation(conversationId: string): Promise<ConversationDetail> {
  const response = await authFetch(`/api/conversations/${conversationId}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to get conversation");
  }
  const raw = (await response.json()) as RawConversationDetail;
  return transformDetail(raw);
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await authFetch(`/api/conversations/${conversationId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to delete conversation");
  }
}
