// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";

export interface MetricsBucket {
  timestamp: string;
  tokens_per_sec: number;
  avg_latency_ms: number;
  total_tokens: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  request_count: number;
  success_rate: number;
}

export interface MetricsResponse {
  model: string | null;
  run_name: string | null;
  period: string;
  start: string;
  end: string;
  buckets: MetricsBucket[];
}

export interface MetricsSummary {
  model: string | null;
  run_name: string | null;
  hours: number;
  tokens_per_sec: number;
  avg_latency_ms: number;
  total_tokens: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  request_count: number;
  success_rate: number;
}

export async function getMetrics(params: {
  model?: string;
  run_name?: string;
  period?: string;
  start?: string;
  end?: string;
}): Promise<MetricsResponse> {
  const qs = new URLSearchParams();
  if (params.model) qs.set("model", params.model);
  if (params.run_name) qs.set("run_name", params.run_name);
  if (params.period) qs.set("period", params.period);
  if (params.start) qs.set("start", params.start);
  if (params.end) qs.set("end", params.end);
  const url = `/api/metrics/${qs.toString() ? `?${qs}` : ""}`;
  const response = await authFetch(url);
  if (!response.ok) throw new Error("Failed to fetch metrics");
  return (await response.json()) as MetricsResponse;
}

export async function getMetricsSummary(params: {
  model?: string;
  run_name?: string;
  hours?: number;
}): Promise<MetricsSummary> {
  const qs = new URLSearchParams();
  if (params.model) qs.set("model", params.model);
  if (params.run_name) qs.set("run_name", params.run_name);
  if (params.hours) qs.set("hours", String(params.hours));
  const url = `/api/metrics/summary${qs.toString() ? `?${qs}` : ""}`;
  const response = await authFetch(url);
  if (!response.ok) throw new Error("Failed to fetch metrics summary");
  return (await response.json()) as MetricsSummary;
}
