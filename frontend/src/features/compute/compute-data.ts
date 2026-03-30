// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

export type NodeWorkload = { name: string; type: string; gpu: number };
export type LocalNode = {
  id: string;
  hostname: string;
  pool: string;
  status: string;
  gpu: { type: string; count: number; used: number; utilization: number } | null;
  cpu: { cores: number; used: number; utilization: number };
  mem: { total: number; used: number; unit: string };
  workloads: NodeWorkload[];
};

export type CloudInstance = {
  id: string;
  provider: string;
  region: string;
  type: string;
  gpu: string;
  status: string;
  workload: string;
  startedAt: string;
  costPerHour: number;
  estimatedTotal: number;
  spotInstance: boolean;
  spotSavings: string;
  autoTerminate: string;
};

export type CloudAccount = {
  provider: string;
  name: string;
  status: string;
  quotaGpu: number;
  usedGpu: number;
  regions: string[];
  monthlyBudget: number;
  monthlySpend: number;
};

export type WorkloadItem = {
  id: string;
  name: string;
  type: string;
  method: string;
  status: string;
  priority: number;
  gpu: string;
  gpuCount: number;
  location: string;
  node: string;
  eta: string;
  startedAt: string | null;
  requestedBy: string;
  project: string;
};

export type ScalingPolicy = {
  id: string;
  name: string;
  enabled: boolean;
  trigger: string;
  action: string;
  maxSpend: string;
  cooldown: string;
  lastTriggered: string;
  triggerCount: number;
};

export type CostDaily = { day: number; value: number };
export type CostByType = { type: string; cost: number; pct: number; color: string };
export type CostByProject = { project: string; cost: number; pct: number; color: string };

// ── Color maps ──────────────────────────────────────────────────────

export const STATUS_COLORS: Record<string, string> = {
  active: "#22C55E", running: "#22C55E", cordoned: "#F59E0B", error: "#EF4444",
  queued: "#6B7280", provisioning: "#06B6D4", connected: "#22C55E", disconnected: "#3A4154",
  completed: "#22C55E", failed: "#EF4444", cancelled: "#6B7280",
};

export const WORKLOAD_COLORS: Record<string, string> = {
  training: "#F59E0B", serving: "#22C55E", eval: "#8B5CF6", system: "#3A4154", idle: "#1A1F2E",
};

export const PROVIDER_COLORS: Record<string, string> = {
  aws: "#FF9900", gcp: "#4285F4", azure: "#0078D4",
};

// ── Mock data ───────────────────────────────────────────────────────

function genSeries(len: number, min: number, max: number) {
  const d: number[] = [];
  let v = min + Math.random() * (max - min);
  for (let i = 0; i < len; i++) {
    v += (Math.random() - 0.48) * (max - min) * 0.08;
    v = Math.max(min, Math.min(max, v));
    d.push(Math.round(v * 100) / 100);
  }
  return d;
}

export const LOCAL_NODES: LocalNode[] = [
  { id: "gpu-node-01", hostname: "gpu-node-01.internal", pool: "training", status: "active", gpu: { type: "H100 80GB", count: 4, used: 4, utilization: 92 }, cpu: { cores: 64, used: 58, utilization: 91 }, mem: { total: 512, used: 420, unit: "Gi" }, workloads: [{ name: "CX SFT Round 4", type: "training", gpu: 4 }] },
  { id: "gpu-node-02", hostname: "gpu-node-02.internal", pool: "training", status: "active", gpu: { type: "A100 80GB", count: 4, used: 4, utilization: 88 }, cpu: { cores: 64, used: 42, utilization: 66 }, mem: { total: 512, used: 380, unit: "Gi" }, workloads: [{ name: "Guard SFT v2 (eval)", type: "eval", gpu: 1 }, { name: "CX Safety Regression", type: "eval", gpu: 1 }, { name: "idle", type: "idle", gpu: 2 }] },
  { id: "gpu-node-03", hostname: "gpu-node-03.internal", pool: "serving", status: "active", gpu: { type: "A100 80GB", count: 4, used: 4, utilization: 74 }, cpu: { cores: 64, used: 38, utilization: 59 }, mem: { total: 512, used: 310, unit: "Gi" }, workloads: [{ name: "llama-3.1-8b-cx (\u00d72)", type: "serving", gpu: 2 }, { name: "deepseek-r1-code (\u00d72)", type: "serving", gpu: 2 }] },
  { id: "gpu-node-04", hostname: "gpu-node-04.internal", pool: "serving", status: "active", gpu: { type: "A100 80GB", count: 4, used: 2, utilization: 45 }, cpu: { cores: 64, used: 22, utilization: 34 }, mem: { total: 256, used: 140, unit: "Gi" }, workloads: [{ name: "deepseek-r1-code (\u00d72)", type: "serving", gpu: 2 }, { name: "idle", type: "idle", gpu: 2 }] },
  { id: "gpu-node-05", hostname: "gpu-node-05.internal", pool: "serving", status: "active", gpu: { type: "H100 80GB", count: 4, used: 4, utilization: 48 }, cpu: { cores: 64, used: 28, utilization: 44 }, mem: { total: 512, used: 280, unit: "Gi" }, workloads: [{ name: "qwen-2.5-72b (\u00d74)", type: "serving", gpu: 4 }] },
  { id: "gpu-node-06", hostname: "gpu-node-06.internal", pool: "training", status: "active", gpu: { type: "A100 80GB", count: 4, used: 0, utilization: 0 }, cpu: { cores: 64, used: 4, utilization: 6 }, mem: { total: 256, used: 18, unit: "Gi" }, workloads: [] },
  { id: "cpu-node-01", hostname: "cpu-node-01.internal", pool: "system", status: "active", gpu: null, cpu: { cores: 32, used: 18, utilization: 56 }, mem: { total: 128, used: 82, unit: "Gi" }, workloads: [{ name: "platform services", type: "system", gpu: 0 }] },
  { id: "gpu-node-07", hostname: "gpu-node-07.internal", pool: "training", status: "cordoned", gpu: { type: "A100 80GB", count: 4, used: 0, utilization: 0 }, cpu: { cores: 64, used: 0, utilization: 0 }, mem: { total: 256, used: 0, unit: "Gi" }, workloads: [] },
];

export const CLOUD_INSTANCES: CloudInstance[] = [
  { id: "sky-a1b2c3", provider: "aws", region: "us-east-1", type: "p4d.24xlarge", gpu: "8\u00d7 A100 80GB", status: "running", workload: "Code RL Phase 2", startedAt: "6h ago", costPerHour: 32.77, estimatedTotal: 196.62, spotInstance: true, spotSavings: "62%", autoTerminate: "2h 14m" },
  { id: "sky-d4e5f6", provider: "aws", region: "us-west-2", type: "p4d.24xlarge", gpu: "8\u00d7 A100 80GB", status: "provisioning", workload: "Queued: Qwen eval suite", startedAt: "2m ago", costPerHour: 32.77, estimatedTotal: 0, spotInstance: true, spotSavings: "58%", autoTerminate: "4h" },
];

export const CLOUD_ACCOUNTS: CloudAccount[] = [
  { provider: "aws", name: "AWS Production", status: "connected", quotaGpu: 64, usedGpu: 16, regions: ["us-east-1", "us-west-2", "eu-west-1"], monthlyBudget: 15000, monthlySpend: 4820 },
  { provider: "gcp", name: "GCP Research", status: "connected", quotaGpu: 32, usedGpu: 0, regions: ["us-central1", "europe-west4"], monthlyBudget: 8000, monthlySpend: 1200 },
  { provider: "azure", name: "Azure (backup)", status: "disconnected", quotaGpu: 0, usedGpu: 0, regions: [], monthlyBudget: 0, monthlySpend: 0 },
];

export const WORKLOAD_QUEUE: WorkloadItem[] = [
  { id: "w-001", name: "CX SFT Round 4", type: "training", method: "SFT", status: "running", priority: 1, gpu: "4\u00d7 H100", gpuCount: 4, location: "local", node: "gpu-node-01", eta: "~1h 30m", startedAt: "2h ago", requestedBy: "A. Kov\u00e1cs", project: "prod-cx" },
  { id: "w-002", name: "Code RL Phase 2", type: "training", method: "GRPO", status: "running", priority: 1, gpu: "8\u00d7 A100", gpuCount: 8, location: "aws", node: "sky-a1b2c3", eta: "~8h", startedAt: "6h ago", requestedBy: "M. Chen", project: "prod-code" },
  { id: "w-003", name: "CX Safety Regression", type: "eval", method: "\u2014", status: "running", priority: 2, gpu: "1\u00d7 A100", gpuCount: 1, location: "local", node: "gpu-node-02", eta: "~18m", startedAt: "12m ago", requestedBy: "A. Kov\u00e1cs", project: "prod-cx" },
  { id: "w-004", name: "llama-3.1-8b-cx", type: "serving", method: "\u2014", status: "running", priority: 0, gpu: "2\u00d7 A100", gpuCount: 2, location: "local", node: "gpu-node-03", eta: "\u2014", startedAt: "14d ago", requestedBy: "system", project: "prod-cx" },
  { id: "w-005", name: "deepseek-r1-code", type: "serving", method: "\u2014", status: "running", priority: 0, gpu: "4\u00d7 A100", gpuCount: 4, location: "local", node: "gpu-node-03/04", eta: "\u2014", startedAt: "8d ago", requestedBy: "system", project: "prod-code" },
  { id: "w-006", name: "qwen-2.5-72b", type: "serving", method: "\u2014", status: "running", priority: 0, gpu: "4\u00d7 H100", gpuCount: 4, location: "local", node: "gpu-node-05", eta: "\u2014", startedAt: "22d ago", requestedBy: "system", project: "staging-da" },
  { id: "w-007", name: "CX DPO Phase 1", type: "training", method: "DPO", status: "queued", priority: 2, gpu: "4\u00d7 H100", gpuCount: 4, location: "local", node: "\u2014", eta: "~1h 30m wait", startedAt: null, requestedBy: "A. Kov\u00e1cs", project: "prod-cx" },
  { id: "w-008", name: "Qwen eval suite", type: "eval", method: "\u2014", status: "provisioning", priority: 3, gpu: "4\u00d7 A100", gpuCount: 4, location: "aws", node: "sky-d4e5f6", eta: "~5m setup", startedAt: null, requestedBy: "R. Silva", project: "staging-da" },
];

export const COST_DAILY: CostDaily[] = genSeries(30, 120, 380).map((v, i) => ({ day: i + 1, value: Math.round(v) }));

export const COST_BY_TYPE: CostByType[] = [
  { type: "Training (SFT)", cost: 2840, pct: 42, color: "#F59E0B" },
  { type: "Training (RL)", cost: 1620, pct: 24, color: "#3B82F6" },
  { type: "Model Serving", cost: 1200, pct: 18, color: "#22C55E" },
  { type: "Evaluations", cost: 680, pct: 10, color: "#8B5CF6" },
  { type: "Data Pipelines", cost: 420, pct: 6, color: "#06B6D4" },
];

export const COST_BY_PROJECT: CostByProject[] = [
  { project: "prod-cx", cost: 3200, pct: 47, color: "#F59E0B" },
  { project: "prod-code", cost: 2400, pct: 35, color: "#3B82F6" },
  { project: "staging-da", cost: 1160, pct: 18, color: "#8B5CF6" },
];

export const SCALING_POLICIES: ScalingPolicy[] = [
  { id: "pol-001", name: "Training Queue Overflow", enabled: true, trigger: "GPU queue depth > 3 AND wait time > 30min", action: "Provision SkyPilot spot instances (up to 8\u00d7 A100)", maxSpend: "$500/job", cooldown: "15 min", lastTriggered: "6h ago", triggerCount: 4 },
  { id: "pol-002", name: "Serving Auto-scale", enabled: true, trigger: "Agent RPS > 150 OR p99 latency > 500ms", action: "Scale serving replicas +1 on local cluster", maxSpend: "\u2014", cooldown: "5 min", lastTriggered: "2h ago", triggerCount: 12 },
  { id: "pol-003", name: "Cloud Auto-terminate", enabled: true, trigger: "Cloud instance idle > 15min OR job completed", action: "Terminate SkyPilot instance immediately", maxSpend: "\u2014", cooldown: "\u2014", lastTriggered: "1h ago", triggerCount: 8 },
  { id: "pol-004", name: "Budget Guard", enabled: true, trigger: "Monthly cloud spend > 80% of budget", action: "Block new cloud provisioning, alert admin", maxSpend: "\u2014", cooldown: "\u2014", lastTriggered: "never", triggerCount: 0 },
  { id: "pol-005", name: "Night Batch Training", enabled: false, trigger: "Weekdays 22:00-06:00 UTC", action: "Release serving GPUs for queued training jobs", maxSpend: "\u2014", cooldown: "\u2014", lastTriggered: "never", triggerCount: 0 },
];
