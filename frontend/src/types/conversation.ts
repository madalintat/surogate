// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

export interface ToolCall {
  id: string;
  type: string;
  function: { name: string; arguments: string };
}

export interface ConversationMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  reasoningContent?: string | null;
  toolCalls?: ToolCall[] | null;
  toolCallId?: string | null;
  timestamp: string | null;
  tokens: number | null;
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
