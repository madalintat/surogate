// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { LucideIcon } from "lucide-react";
import { Layers, Database, Bot, Zap, FileText, Settings2, Type, FileCode, Folder } from "lucide-react";

export interface RepoCommit {
  hash: string;
  message: string;
  author: string;
  date: string;
  tag?: string;
}

export interface RepoFile {
  name: string;
  size: string;
  type: "weights" | "config" | "tokenizer" | "doc" | "data" | "code" | "dir";
}

export interface RepoCard {
  metrics: Record<string, string | number> | null;
  training: Record<string, string | number> | null;
}

export type RepoType = "model" | "dataset" | "agent" | "skill";
export type Visibility = "public" | "internal" | "private";

export interface RepoItem {
  id: string;
  type: RepoType;
  name: string;
  displayName: string;
  description: string;
  version: string;
  project: string;
  projectColor: string;
  visibility: Visibility;
  author: string;
  updatedAt: string;
  createdAt: string;
  downloads: number;
  likes: number;
  tags: string[];
  size: string;
  format: string;
  license: string;
  baseModel: string | null;
  trainingRun: string | null;
  serving: boolean;
  commits: RepoCommit[];
  files: RepoFile[];
  card: RepoCard;
}

export const TYPE_META: Record<RepoType, { icon: LucideIcon; color: string; label: string; plural: string }> = {
  model: { icon: Layers, color: "#3B82F6", label: "Model", plural: "Models" },
  dataset: { icon: Database, color: "#22C55E", label: "Dataset", plural: "Datasets" },
  agent: { icon: Bot, color: "#F59E0B", label: "Agent Config", plural: "Agent Configs" },
  skill: { icon: Zap, color: "#8B5CF6", label: "Skill", plural: "Skills" },
};

export const VIS_COLORS: Record<Visibility, { bg: string; fg: string }> = {
  public: { bg: "#22C55E12", fg: "#22C55E" },
  internal: { bg: "#3B82F612", fg: "#3B82F6" },
  private: { bg: "#EF444412", fg: "#EF4444" },
};

export const FILE_ICONS: Record<string, LucideIcon> = {
  weights: Layers,
  config: Settings2,
  tokenizer: Type,
  doc: FileText,
  data: Database,
  code: FileCode,
  dir: Folder,
};

export const REPO_ITEMS: RepoItem[] = [
  {
    id: "models/llama-3.1-8b-cx/v4",
    type: "model",
    name: "llama-3.1-8b-cx",
    displayName: "Llama 3.1 8B — CX Fine-tune v4",
    description: "Production fine-tune for customer experience. SFT+DPO on 2,840 curated support conversations. LoRA rank 64, AWQ 4-bit quantized.",
    version: "v4",
    project: "prod-cx",
    projectColor: "#F59E0B",
    visibility: "internal",
    author: "A. Kovács",
    updatedAt: "2h ago",
    createdAt: "2026-03-25",
    downloads: 42,
    likes: 8,
    tags: ["llama-3.1", "8b", "fine-tuned", "cx", "sft", "dpo", "awq", "production"],
    size: "4.2 GB",
    format: "safetensors",
    license: "Llama 3.1 Community",
    baseModel: "meta-llama/Llama-3.1-8B-Instruct",
    trainingRun: "ft-0042",
    serving: true,
    commits: [
      { hash: "a3f8c21", message: "Publish v4 weights (SFT+DPO, AWQ quantized)", author: "A. Kovács", date: "2h ago", tag: "v4" },
      { hash: "b7e4d09", message: "Add model card and evaluation results", author: "A. Kovács", date: "3h ago" },
      { hash: "c1a9f33", message: "Upload LoRA adapters (pre-merge)", author: "A. Kovács", date: "4h ago" },
    ],
    files: [
      { name: "model.safetensors", size: "4.1 GB", type: "weights" },
      { name: "config.json", size: "1.2 KB", type: "config" },
      { name: "tokenizer.json", size: "1.8 MB", type: "tokenizer" },
      { name: "tokenizer_config.json", size: "420 B", type: "config" },
      { name: "special_tokens_map.json", size: "280 B", type: "config" },
      { name: "README.md", size: "3.4 KB", type: "doc" },
      { name: "eval_results.json", size: "2.1 KB", type: "data" },
    ],
    card: {
      metrics: { gsm8k: 82.4, mmlu: 68.2, "cx-quality": 4.6, mtbench: 7.8, toxigen: 96.2 },
      training: { method: "SFT + DPO", dataset: "cx-convos-v5 + cx-dpo-pairs-v1", epochs: 3, lr: "2e-5", lora: "rank 64, alpha 128" },
    },
  },
  {
    id: "models/llama-3.1-8b-cx/v3",
    type: "model",
    name: "llama-3.1-8b-cx",
    displayName: "Llama 3.1 8B — CX Fine-tune v3",
    description: "Previous production fine-tune. SFT only on v4 dataset. Superseded by v4 which adds DPO training.",
    version: "v3",
    project: "prod-cx",
    projectColor: "#F59E0B",
    visibility: "internal",
    author: "A. Kovács",
    updatedAt: "1w ago",
    createdAt: "2026-03-18",
    downloads: 128,
    likes: 12,
    tags: ["llama-3.1", "8b", "fine-tuned", "cx", "sft", "previous"],
    size: "4.2 GB",
    format: "safetensors",
    license: "Llama 3.1 Community",
    baseModel: "meta-llama/Llama-3.1-8B-Instruct",
    trainingRun: "ft-0040",
    serving: false,
    commits: [
      { hash: "d5b2e77", message: "Publish v3 weights (SFT, AWQ)", author: "A. Kovács", date: "1w ago", tag: "v3" },
    ],
    files: [
      { name: "model.safetensors", size: "4.1 GB", type: "weights" },
      { name: "config.json", size: "1.2 KB", type: "config" },
      { name: "README.md", size: "2.8 KB", type: "doc" },
    ],
    card: {
      metrics: { gsm8k: 78.1, mmlu: 67.8, "cx-quality": 4.1, mtbench: 7.4 },
      training: { method: "SFT", dataset: "cx-convos-v4", epochs: 3, lr: "2e-5", lora: "rank 64, alpha 128" },
    },
  },
  {
    id: "models/deepseek-r1-code/v1",
    type: "model",
    name: "deepseek-r1-code",
    displayName: "DeepSeek R1 — Code",
    description: "Base DeepSeek R1 model for code generation. FP16, 128K context. Imported from HuggingFace Hub.",
    version: "v1",
    project: "prod-code",
    projectColor: "#3B82F6",
    visibility: "internal",
    author: "M. Chen",
    updatedAt: "1w ago",
    createdAt: "2026-03-21",
    downloads: 36,
    likes: 15,
    tags: ["deepseek", "r1", "70b", "code", "base", "fp16", "128k-context"],
    size: "140 GB",
    format: "safetensors",
    license: "DeepSeek License",
    baseModel: "deepseek-ai/DeepSeek-R1",
    trainingRun: null,
    serving: true,
    commits: [
      { hash: "f2a8b11", message: "Import from HuggingFace Hub", author: "M. Chen", date: "1w ago", tag: "v1" },
    ],
    files: [
      { name: "model-00001-of-00030.safetensors", size: "4.7 GB", type: "weights" },
      { name: "config.json", size: "1.8 KB", type: "config" },
      { name: "README.md", size: "5.2 KB", type: "doc" },
    ],
    card: {
      metrics: { humaneval: 91.5, mbpp: 86.2, gsm8k: 94.1, mmlu: 82.4, mtbench: 8.6 },
      training: null,
    },
  },
  {
    id: "models/guard-3b/v2",
    type: "model",
    name: "guard-3b",
    displayName: "LlamaGuard 3B v2",
    description: "Safety classifier fine-tuned on expanded demographic labels. 13 demographic groups, improved recall.",
    version: "v2",
    project: "staging-da",
    projectColor: "#8B5CF6",
    visibility: "internal",
    author: "A. Kovács",
    updatedAt: "2d ago",
    createdAt: "2026-03-26",
    downloads: 8,
    likes: 3,
    tags: ["llamaguard", "3b", "safety", "classifier", "fine-tuned"],
    size: "6.4 GB",
    format: "safetensors",
    license: "Llama 3.1 Community",
    baseModel: "meta-llama/Llama-Guard-3-1B",
    trainingRun: "ft-0040b",
    serving: false,
    commits: [
      { hash: "p6o1q00", message: "Publish v2 weights (extended demographics)", author: "A. Kovács", date: "2d ago", tag: "v2" },
    ],
    files: [
      { name: "model.safetensors", size: "6.2 GB", type: "weights" },
      { name: "config.json", size: "980 B", type: "config" },
      { name: "README.md", size: "2.1 KB", type: "doc" },
    ],
    card: {
      metrics: { toxigen: 97.8, truthfulqa: 61.2 },
      training: { method: "SFT", dataset: "safety-labels-v2", epochs: 3, lr: "5e-5", lora: "rank 16, alpha 32" },
    },
  },
  {
    id: "datasets/cx-convos-v5",
    type: "dataset",
    name: "cx-convos-v5",
    displayName: "CX Conversations v5",
    description: "2,840 curated SFT samples from cx-support-v3 production. De-identified, quality-scored, sentiment-labeled.",
    version: "v5",
    project: "prod-cx",
    projectColor: "#F59E0B",
    visibility: "internal",
    author: "A. Kovács",
    updatedAt: "2h ago",
    createdAt: "2026-03-25",
    downloads: 18,
    likes: 6,
    tags: ["cx", "sft", "conversations", "production", "annotated", "2840-samples"],
    size: "48 MB",
    format: "JSONL",
    license: "Internal",
    baseModel: null,
    trainingRun: null,
    serving: false,
    commits: [
      { hash: "k8m2n44", message: "Add 340 new annotated conversations", author: "A. Kovács", date: "2h ago", tag: "v5" },
      { hash: "j6l0p33", message: "Apply de-identification pipeline", author: "A. Kovács", date: "1w ago" },
    ],
    files: [
      { name: "train.jsonl", size: "42 MB", type: "data" },
      { name: "val.jsonl", size: "5.8 MB", type: "data" },
      { name: "metadata.json", size: "12 KB", type: "config" },
      { name: "README.md", size: "1.8 KB", type: "doc" },
    ],
    card: {
      metrics: { samples: 2840, avgTokens: 1480, avgTurns: 7.2 },
      training: null,
    },
  },
  {
    id: "datasets/code-trajectories-v2",
    type: "dataset",
    name: "code-trajectories-v2",
    displayName: "Code Trajectories v2",
    description: "1,420 GRPO trajectories with test-outcome rewards from code-assist-v2. Multi-turn with tool call traces.",
    version: "v2",
    project: "prod-code",
    projectColor: "#3B82F6",
    visibility: "internal",
    author: "M. Chen",
    updatedAt: "1d ago",
    createdAt: "2026-03-20",
    downloads: 12,
    likes: 9,
    tags: ["code", "grpo", "trajectories", "rewards", "tool-use", "1420-samples"],
    size: "142 MB",
    format: "JSONL",
    license: "Internal",
    baseModel: null,
    trainingRun: null,
    serving: false,
    commits: [
      { hash: "r4t6u88", message: "Add reward signals from test outcomes", author: "M. Chen", date: "1d ago", tag: "v2" },
    ],
    files: [
      { name: "trajectories.jsonl", size: "138 MB", type: "data" },
      { name: "rewards.json", size: "3.2 MB", type: "data" },
      { name: "README.md", size: "2.4 KB", type: "doc" },
    ],
    card: {
      metrics: { samples: 1420, avgTokens: 9014, avgTurns: 18.4 },
      training: null,
    },
  },
  {
    id: "agents/cx-support-v3",
    type: "agent",
    name: "cx-support-v3",
    displayName: "CX Support Agent v3",
    description: "Full agent deployment config: system prompt v14, 5 skills, 2 MCP servers, escalation routing, sentiment guardrail.",
    version: "v3.2.1",
    project: "prod-cx",
    projectColor: "#F59E0B",
    visibility: "internal",
    author: "A. Kovács",
    updatedAt: "2h ago",
    createdAt: "2025-09-14",
    downloads: 6,
    likes: 4,
    tags: ["agent", "cx", "production", "support", "multi-skill"],
    size: "24 KB",
    format: "YAML",
    license: "Internal",
    baseModel: null,
    trainingRun: null,
    serving: true,
    commits: [
      { hash: "a3f8c21", message: "Tune escalation threshold to 0.85", author: "A. Kovács", date: "2h ago", tag: "v3.2.1" },
      { hash: "b7e4d09", message: "Add sentiment-guard skill", author: "M. Chen", date: "2d ago", tag: "v3.2.0" },
      { hash: "c1a9f33", message: "Update system prompt v14", author: "A. Kovács", date: "1w ago" },
      { hash: "d5b2e77", message: "Switch to llama-3.1-8b-cx v4 model", author: "A. Kovács", date: "1w ago" },
    ],
    files: [
      { name: "agent.yaml", size: "8.2 KB", type: "config" },
      { name: "system_prompt.txt", size: "2.4 KB", type: "doc" },
      { name: "skills.yaml", size: "3.1 KB", type: "config" },
      { name: "mcp_servers.yaml", size: "1.2 KB", type: "config" },
      { name: "env.yaml", size: "680 B", type: "config" },
      { name: "README.md", size: "4.8 KB", type: "doc" },
    ],
    card: {
      metrics: { skills: 5, mcpServers: 2, conversations24h: 1847, satisfaction: "94%" },
      training: null,
    },
  },
  {
    id: "agents/code-assist-v2",
    type: "agent",
    name: "code-assist-v2",
    displayName: "Code Assistant v2",
    description: "Developer productivity agent config. 4 skills including repo-indexer and code-executor. 128K context.",
    version: "v2.7.0",
    project: "prod-code",
    projectColor: "#3B82F6",
    visibility: "internal",
    author: "M. Chen",
    updatedAt: "6h ago",
    createdAt: "2025-11-02",
    downloads: 4,
    likes: 7,
    tags: ["agent", "code", "production", "developer-tools"],
    size: "18 KB",
    format: "YAML",
    license: "Internal",
    baseModel: null,
    trainingRun: null,
    serving: true,
    commits: [
      { hash: "f2a8b11", message: "Upgrade context window to 128K", author: "M. Chen", date: "6h ago", tag: "v2.7.0" },
    ],
    files: [
      { name: "agent.yaml", size: "6.8 KB", type: "config" },
      { name: "system_prompt.txt", size: "1.9 KB", type: "doc" },
      { name: "skills.yaml", size: "2.4 KB", type: "config" },
      { name: "README.md", size: "3.2 KB", type: "doc" },
    ],
    card: {
      metrics: { skills: 4, mcpServers: 2, conversations24h: 412, satisfaction: "97%" },
      training: null,
    },
  },
  {
    id: "skills/kb-search/v3.0.1",
    type: "skill",
    name: "kb-search",
    displayName: "Knowledge Base Search v3",
    description: "Hybrid semantic search with BGE embeddings, sparse retrieval, cross-encoder reranking, and citation extraction.",
    version: "v3.0.1",
    project: "prod-cx",
    projectColor: "#F59E0B",
    visibility: "public",
    author: "A. Kovács",
    updatedAt: "3d ago",
    createdAt: "2025-06-20",
    downloads: 84,
    likes: 22,
    tags: ["skill", "rag", "search", "embeddings", "hybrid", "reranking"],
    size: "12 KB",
    format: "Python",
    license: "Apache 2.0",
    baseModel: null,
    trainingRun: null,
    serving: false,
    commits: [
      { hash: "s2v4w66", message: "Fix citation extraction for PDFs", author: "A. Kovács", date: "3d ago", tag: "v3.0.1" },
      { hash: "t4x6y77", message: "Add hybrid search + reranking", author: "A. Kovács", date: "2w ago", tag: "v3.0.0" },
    ],
    files: [
      { name: "skill.py", size: "4.2 KB", type: "code" },
      { name: "schema.json", size: "1.8 KB", type: "config" },
      { name: "requirements.txt", size: "280 B", type: "config" },
      { name: "README.md", size: "3.6 KB", type: "doc" },
      { name: "tests/", size: "—", type: "dir" },
    ],
    card: {
      metrics: { calls24h: 8940, avgLatency: "85ms", usedByAgents: 1 },
      training: null,
    },
  },
  {
    id: "skills/escalation-router/v1.0.0",
    type: "skill",
    name: "escalation-router",
    displayName: "Escalation Router v1",
    description: "Multi-step workflow: sentiment evaluation, uncertainty detection, priority assignment, and human handoff with context summary.",
    version: "v1.0.0",
    project: "prod-cx",
    projectColor: "#F59E0B",
    visibility: "public",
    author: "A. Kovács",
    updatedAt: "2w ago",
    createdAt: "2025-12-01",
    downloads: 56,
    likes: 14,
    tags: ["skill", "workflow", "escalation", "routing", "human-handoff"],
    size: "8 KB",
    format: "Python",
    license: "Apache 2.0",
    baseModel: null,
    trainingRun: null,
    serving: false,
    commits: [
      { hash: "u6z8a99", message: "GA release with priority levels", author: "A. Kovács", date: "2w ago", tag: "v1.0.0" },
    ],
    files: [
      { name: "skill.py", size: "3.8 KB", type: "code" },
      { name: "schema.json", size: "1.4 KB", type: "config" },
      { name: "README.md", size: "2.8 KB", type: "doc" },
    ],
    card: {
      metrics: { calls24h: 2200, avgLatency: "25ms", usedByAgents: 1 },
      training: null,
    },
  },
];
