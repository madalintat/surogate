// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sparkline } from "@/components/ui/sparkline";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusDot } from "@/components/ui/status-dot";
import type { Status } from "@/components/ui/status-dot";
import { PageHeader } from "@/components/page-header";
import { cn } from "@/utils/cn";

/* ── sample data ── */


const AGENTS_DATA = [
  { name: "cx-support-v3", version: "3.2.1", status: "running" as Status, replicas: "3/3", cpu: "42%", mem: "1.8Gi", rps: 124, p99: "320ms", model: "llama-3.1-8b-cx" },
  { name: "code-assist-v2", version: "2.7.0", status: "running" as Status, replicas: "2/2", cpu: "67%", mem: "3.2Gi", rps: 89, p99: "890ms", model: "deepseek-r1-code" },
  { name: "data-analyst-v1", version: "1.4.0-rc2", status: "deploying" as Status, replicas: "1/2", cpu: "23%", mem: "0.9Gi", rps: 12, p99: "1.2s", model: "qwen-2.5-72b" },
  { name: "onboarding-bot", version: "1.0.3", status: "running" as Status, replicas: "1/1", cpu: "8%", mem: "0.4Gi", rps: 5, p99: "210ms", model: "llama-3.1-8b-cx" },
  { name: "safety-reviewer", version: "0.9.1", status: "error" as Status, replicas: "0/1", cpu: "0%", mem: "0Gi", rps: 0, p99: "—", model: "guard-3b" },
];

const MODELS_DATA = [
  { name: "llama-3.1-8b-cx", base: "Llama 3.1 8B", type: "Fine-tuned", status: "serving" as Status, gpu: "2× A100", tps: 1840 },
  { name: "deepseek-r1-code", base: "DeepSeek R1", type: "Base", status: "serving" as Status, gpu: "4× A100", tps: 920 },
  { name: "qwen-2.5-72b", base: "Qwen 2.5 72B", type: "Base", status: "serving" as Status, gpu: "4× H100", tps: 1100 },
  { name: "guard-3b", base: "LlamaGuard 3B", type: "Fine-tuned", status: "error" as Status, gpu: "1× A100", tps: 0 },
];

const TRAINING_JOBS = [
  { id: "ft-0042", name: "CX SFT Round 4", type: "SFT", status: "running" as Status, progress: 67, epoch: "2/3", loss: "0.847", compute: "local", gpu: "4× H100" },
  { id: "ft-0041", name: "Code RL Phase 2", type: "GRPO", status: "running" as Status, progress: 34, epoch: "1/5", loss: "1.203", compute: "aws", gpu: "8× A100" },
  { id: "ft-0040", name: "Guard classifier v2", type: "SFT", status: "completed" as Status, progress: 100, epoch: "3/3", loss: "0.312", compute: "local", gpu: "1× A100" },
];

const CONVERSATIONS_RECENT = [
  { id: "c-9821", agent: "cx-support-v3", user: "user_8472", preview: "I need to change my subscription plan to...", tokens: 2340, turns: 8, sentiment: "positive", flagged: false, time: "2m ago" },
  { id: "c-9820", agent: "cx-support-v3", user: "user_3109", preview: "Why was I charged twice for the same...", tokens: 4120, turns: 14, sentiment: "negative", flagged: true, time: "5m ago" },
  { id: "c-9819", agent: "code-assist-v2", user: "dev_riley", preview: "Help me refactor this React component to use...", tokens: 8900, turns: 22, sentiment: "positive", flagged: false, time: "8m ago" },
  { id: "c-9818", agent: "data-analyst-v1", user: "analyst_jen", preview: "Generate a quarterly revenue breakdown by...", tokens: 3200, turns: 6, sentiment: "neutral", flagged: false, time: "12m ago" },
  { id: "c-9817", agent: "cx-support-v3", user: "user_0091", preview: "The agent couldn't understand my request about...", tokens: 1890, turns: 11, sentiment: "negative", flagged: true, time: "15m ago" },
];

const QUICK_ACTIONS = [
  { icon: "⬡", label: "Deploy Agent", desc: "From template or config" },
  { icon: "◇", label: "Serve Model", desc: "Deploy LLM to cluster" },
  { icon: "⚡", label: "New Skill", desc: "Create reusable skill" },
  { icon: "◬", label: "Start Training", desc: "SFT or RL job" },
  { icon: "◈", label: "Run Eval", desc: "Benchmark or custom" },
  { icon: "▤", label: "Import Dataset", desc: "From conversations" },
];

const ACTIVITY = [
  { time: "2m", icon: "⬡", text: "data-analyst-v1 scaling to 2 replicas", color: "#F59E0B" },
  { time: "8m", icon: "◬", text: "CX SFT Round 4 — epoch 2 started", color: "#8B5CF6" },
  { time: "14m", icon: "◈", text: "GSM8K eval completed: 82.4%", color: "#3B82F6" },
  { time: "22m", icon: "⊕", text: "llama-3.1-8b-cx-v4 pushed to Hub", color: "#22C55E" },
  { time: "31m", icon: "⚡", text: "New skill order-lookup added to CX", color: "#F59E0B" },
  { time: "45m", icon: "⊡", text: "2 conversations flagged for review", color: "#EF4444" },
  { time: "1h", icon: "◇", text: "guard-3b OOM — restarting with 16Gi", color: "#EF4444" },
];

const CLUSTER_METRICS = [
  { label: "GPU Utilization", value: "78%", max: 100, current: 78, color: "#F59E0B" },
  { label: "CPU Utilization", value: "42%", max: 100, current: 42, color: "#3B82F6" },
  { label: "Memory", value: "186 / 256 Gi", max: 256, current: 186, color: "#8B5CF6" },
  { label: "GPU Nodes", value: "6 / 8 active", max: 8, current: 6, color: "#22C55E" },
];

const CLUSTER_STATS = [
  { label: "Namespaces", value: "3" },
  { label: "Pods Running", value: "24" },
  { label: "Services", value: "12" },
  { label: "PVCs", value: "18" },
];

const SPARK_DATA = {
  agents: [12, 18, 14, 22, 19, 28, 25, 32, 29, 35, 31, 38, 42, 39, 45],
  latency: [320, 310, 340, 290, 300, 280, 310, 290, 270, 300, 280, 260, 270, 250, 260],
  tokens: [1.2, 1.4, 1.1, 1.6, 1.3, 1.8, 2.0, 1.7, 2.1, 2.4, 2.2, 2.6, 2.3, 2.8, 3.1],
  success: [89, 91, 88, 92, 90, 93, 91, 94, 92, 95, 93, 96, 94, 95, 97],
};

const METRIC_CARDS = [
  { label: "Active Agents", value: "4", sub: "+1 deploying", spark: SPARK_DATA.agents, color: "#22C55E" },
  { label: "Avg Response Time", value: "284ms", sub: "p99 across all", spark: SPARK_DATA.latency, color: "#3B82F6" },
  { label: "Tokens Today", value: "3.1M", sub: "↑ 12% from yesterday", spark: SPARK_DATA.tokens, color: "#F59E0B" },
  { label: "Success Rate", value: "97.2%", sub: "Last 24 hours", spark: SPARK_DATA.success, color: "#8B5CF6" },
];

const AGENT_COLUMNS = ["Agent", "Version", "Status", "Replicas", "CPU", "MEM", "RPS", "P99", "Model"] as const;

const SENTIMENT_COLORS: Record<string, string> = {
  positive: "#22C55E",
  negative: "#EF4444",
  neutral: "var(--muted-foreground)",
};

/* ── page ── */

export function StudioPage() {
  const now = new Date();

  return (
    <div className="flex-1 overflow-auto bg-background">
      <PageHeader
        title="Dashboard"
        subtitle={
          <>
            {now.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
            {" · "}
            <span className="text-primary">All systems operational</span>
          </>
        }
      />

      <div className="p-7 pb-10">
        {/* metric cards */}
        <div className="grid grid-cols-4 gap-3 mb-5">
          {METRIC_CARDS.map((m) => (
            <Card key={m.label} className="px-4 py-3.5 flex justify-between items-end">
              <div>
                <div className=" text-muted-foreground uppercase tracking-wider mb-1.5 font-display">
                  {m.label}
                </div>
                <div className="text-[22px] font-bold text-foreground tracking-tight">
                  {m.value}
                </div>
                <div className=" mt-0.5" style={{ color: m.color }}>
                  {m.sub}
                </div>
              </div>
              <Sparkline data={m.spark} color={m.color} />
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-[1fr_320px] gap-4">
          {/* left column */}
          <div className="flex flex-col gap-4">
            {/* agents table */}
            <Card>
              <CardHeader
                icon="⬡"
                iconColor="#F59E0B"
                title="Deployed Agents"
                badge={<Badge>5 agents</Badge>}
                actions={
                  <button
                    type="button"
                    className="bg-linear-to-br from-amber-500 to-amber-600 border-none rounded-[5px] px-3 py-[5px] font-semibold text-primary-foreground cursor-pointer font-display"
                  >
                    + Deploy
                  </button>
                }
              />
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b border-line">
                      {AGENT_COLUMNS.map((h) => (
                        <th
                          key={h}
                          className="px-3 py-2 text-left text-[9px] font-medium text-faint uppercase tracking-widest font-display"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {AGENTS_DATA.map((a) => (
                      <tr
                        key={a.name}
                        className="border-b border-input cursor-pointer transition-colors duration-100 hover:bg-input"
                      >
                        <td className="px-3 py-2.5 text-foreground font-medium">{a.name}</td>
                        <td className="px-3 py-2.5">
                          <code className="text-subtle bg-line px-1.5 py-[2px] rounded">{a.version}</code>
                        </td>
                        <td className="px-3 py-2.5">
                          <span className="flex items-center">
                            <StatusDot status={a.status} />
                            <span className={cn(
                              a.status === "error" && "text-destructive",
                              a.status === "deploying" && "text-primary",
                              a.status !== "error" && a.status !== "deploying" && "text-subtle",
                            )}>
                              {a.status}
                            </span>
                          </span>
                        </td>
                        <td className="px-3 py-2.5  text-subtle">{a.replicas}</td>
                        <td className="px-3 py-2.5  text-subtle">{a.cpu}</td>
                        <td className="px-3 py-2.5  text-subtle">{a.mem}</td>
                        <td className="px-3 py-2.5  text-foreground/80 font-medium">{a.rps}</td>
                        <td className={cn("px-3 py-2.5 ", a.p99 === "—" ? "text-faint" : "text-subtle")}>{a.p99}</td>
                        <td className="px-3 py-2.5 text-muted-foreground">{a.model}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* models + training */}
            <div className="grid grid-cols-2 gap-4">
              {/* models */}
              <Card>
                <CardHeader icon="◇" iconColor="#3B82F6" title="Serving Models" />
                {MODELS_DATA.map((m) => (
                  <div
                    key={m.name}
                    className="px-4 py-2.5 border-b border-input flex items-center gap-2.5"
                  >
                    <StatusDot status={m.status} />
                    <div className="flex-1 min-w-0">
                      <div className=" text-foreground font-medium truncate">{m.name}</div>
                      <div className="text-[9px] text-faint mt-px">{m.base} · {m.type} · {m.gpu}</div>
                    </div>
                    <div className="text-right">
                      <div className={cn(" font-medium", m.tps > 0 ? "text-foreground/80" : "text-faint")}>
                        {m.tps > 0 ? m.tps : "—"}
                      </div>
                      <div className="text-[8px] text-faint">tok/s</div>
                    </div>
                  </div>
                ))}
              </Card>

              {/* training */}
              <Card>
                <CardHeader icon="◬" iconColor="#8B5CF6" title="Training Jobs" />
                {TRAINING_JOBS.map((j) => (
                  <div key={j.id} className="px-4 py-2.5 border-b border-input">
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-1.5">
                        <StatusDot status={j.status} />
                        <span className=" text-foreground font-medium">{j.name}</span>
                        <span
                          className={cn(
                            "text-[8px] px-1 py-px rounded",
                            j.type === "SFT" ? "bg-accent text-subtle" : "bg-violet-950 text-violet-400",
                          )}
                        >
                          {j.type}
                        </span>
                      </div>
                      <span className="text-muted-foreground">
                        {j.compute === "aws" ? "☁ AWS" : "⊞ Local"}
                      </span>
                    </div>
                    <ProgressBar
                      value={j.progress}
                      color={j.status === "completed" ? "#3B82F6" : j.type === "GRPO" ? "#8B5CF6" : "#F59E0B"}
                      animated
                    />
                    <div className="flex gap-3 text-[9px] text-faint mt-1">
                      <span>Epoch {j.epoch}</span>
                      <span>Loss: {j.loss}</span>
                      <span>{j.gpu}</span>
                    </div>
                  </div>
                ))}
              </Card>
            </div>

            {/* recent conversations */}
            <Card>
              <CardHeader
                icon="⊡"
                iconColor="#22C55E"
                title="Recent Conversations"
                badge={<Badge variant="danger">2 flagged</Badge>}
                actions={
                  <div className="flex gap-1.5">
                    {["All", "Flagged", "Negative"].map((f) => (
                      <button
                        key={f}
                        type="button"
                        className={cn(
                          "px-2 py-[3px] rounded border border-border cursor-pointer font-display transition-colors",
                          f === "All"
                            ? "bg-accent text-foreground/80"
                            : "bg-transparent text-faint hover:text-subtle",
                        )}
                      >
                        {f}
                      </button>
                    ))}
                  </div>
                }
              />
              {CONVERSATIONS_RECENT.map((c) => (
                <div
                  key={c.id}
                  className="px-4 py-2.5 border-b border-input flex items-center gap-3 cursor-pointer transition-colors duration-100 hover:bg-input"
                >
                  <div
                    className="w-1.5 h-1.5 rounded-full shrink-0"
                    style={{ background: SENTIMENT_COLORS[c.sentiment] }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <span className=" text-foreground/80 font-medium">{c.agent}</span>
                      <span className="text-[9px] text-faint">→</span>
                      <span className="text-muted-foreground">{c.user}</span>
                      {c.flagged && <Badge variant="danger">flagged</Badge>}
                    </div>
                    <div className=" text-subtle truncate">{c.preview}</div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-muted-foreground">{c.tokens} tok · {c.turns} turns</div>
                    <div className="text-[9px] text-faint">{c.time}</div>
                  </div>
                </div>
              ))}
            </Card>
          </div>

          {/* right column */}
          <div className="flex flex-col gap-4">
            {/* quick actions */}
            <Card>
              <CardHeader icon="◫" iconColor="#F59E0B" title="Quick Actions" />
              <div className="p-2 grid grid-cols-2 gap-1.5">
                {QUICK_ACTIONS.map((a) => (
                  <button
                    key={a.label}
                    type="button"
                    className="flex flex-col items-start px-3 py-2.5 rounded-md border border-transparent bg-transparent cursor-pointer text-left transition-all duration-150 hover:border-border hover:bg-input"
                  >
                    <span className="text-base mb-1">{a.icon}</span>
                    <span className=" text-foreground font-medium font-display">{a.label}</span>
                    <span className="text-[9px] text-faint">{a.desc}</span>
                  </button>
                ))}
              </div>
            </Card>

            {/* cluster health */}
            <Card>
              <CardHeader icon="⊞" iconColor="#22C55E" title="Cluster Health" />
              <div className="px-4 py-3">
                {CLUSTER_METRICS.map((r, i) => (
                  <div key={r.label} className={i < CLUSTER_METRICS.length - 1 ? "mb-3.5" : ""}>
                    <div className="flex justify-between mb-1.5 font-display">
                      <span className="text-subtle">{r.label}</span>
                      <span className="text-foreground/80 font-medium">{r.value}</span>
                    </div>
                    <ProgressBar value={r.current} max={r.max} color={r.color} />
                  </div>
                ))}
              </div>
              <div className="px-4 py-2 pb-3 grid grid-cols-2 gap-2">
                {CLUSTER_STATS.map((s) => (
                  <div key={s.label} className="flex justify-between">
                    <span className="text-faint font-display">{s.label}</span>
                    <span className="text-subtle font-medium">{s.value}</span>
                  </div>
                ))}
              </div>
            </Card>

            {/* activity feed */}
            <Card>
              <CardHeader icon="⊙" iconColor="var(--subtle)" title="Activity" />
              <div className="py-1">
                {ACTIVITY.map((e) => (
                  <div
                    key={e.time + e.text}
                    className="px-4 py-[7px] flex items-start gap-2.5  cursor-pointer transition-colors duration-100 hover:bg-input"
                  >
                    <span className="text-faint text-[9px] w-6 text-right shrink-0 mt-0.5 font-display">
                      {e.time}
                    </span>
                    <span className=" shrink-0" style={{ color: e.color }}>
                      {e.icon}
                    </span>
                    <span className="text-subtle">{e.text}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
