// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

// ── Types ───────────────────────────────────────────────────────

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  reasoningContent?: string | null;
  timestamp: string | null;
  tokens: number;
  latency?: number;
}

export interface ConversationTurn {
  id: string;
  parentHash: string | null;
  stateHash: string;
  tailHash: string;
  model: string;
  isStreaming: boolean;
  promptTokens: number | null;
  completionTokens: number | null;
  totalTokens: number | null;
  latencyMs: number | null;
  compacted: boolean;
  createdAt: string | null;
}

/** Summary returned by the list endpoint. */
export interface Conversation {
  id: string;
  model: string;
  projectName: string;
  runName: string;
  turnCount: number;
  tokensIn: number;
  tokensOut: number;
  totalTokens: number;
  avgLatencyMs: number | null;
  startedAt: string | null;
  lastTurnAt: string | null;
  hasCompaction: boolean;
  time: string;
  duration: string;
}

/** Full detail returned by the detail endpoint. */
export interface ConversationDetail extends Conversation {
  systemPrompt: string | null;
  messages: ConversationMessage[];
  turns: ConversationTurn[];
}

// ── Constants ──────────────────────────────────────────────────

export const SENTIMENT_COLORS: Record<string, string> = {
  positive: "#22C55E",
  neutral: "#6B7585",
  negative: "#EF4444",
};

export const ANNOTATION_STYLES: Record<
  string,
  { bg: string; fg: string; border: string; label: string }
> = {
  skill_gap: {
    bg: "#F59E0B12",
    fg: "#F59E0B",
    border: "#F59E0B30",
    label: "SKILL GAP",
  },
  trajectory_correction: {
    bg: "#3B82F612",
    fg: "#3B82F6",
    border: "#3B82F630",
    label: "TRAJECTORY",
  },
  quality_issue: {
    bg: "#EF444412",
    fg: "#EF4444",
    border: "#EF444430",
    label: "QUALITY",
  },
};

