// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusDot } from "@/components/ui/status-dot";
import {
  LOCAL_NODES, CLOUD_INSTANCES, CLOUD_ACCOUNTS, WORKLOAD_QUEUE,
  WORKLOAD_COLORS, PROVIDER_COLORS,
} from "./compute-data";

export function OverviewTab() {
  const totalLocalGpu = LOCAL_NODES.reduce((s, n) => s + (n.gpu?.count || 0), 0);
  const usedLocalGpu = LOCAL_NODES.reduce((s, n) => s + (n.gpu?.used || 0), 0);
  const totalCloudGpu = CLOUD_INSTANCES.reduce((s, c) => s + parseInt(c.gpu), 0);
  const cloudHourlyCost = CLOUD_INSTANCES.filter(c => c.status === "running").reduce((s, c) => s + c.costPerHour, 0);
  const monthlySpend = CLOUD_ACCOUNTS.reduce((s, a) => s + a.monthlySpend, 0);
  const monthlyBudget = CLOUD_ACCOUNTS.reduce((s, a) => s + a.monthlyBudget, 0);

  const kpis = [
    { label: "Local GPUs", value: `${usedLocalGpu}/${totalLocalGpu}`, sub: `${LOCAL_NODES.filter(n => n.status === "active").length} nodes` },
    { label: "Cloud GPUs", value: totalCloudGpu, sub: `${CLOUD_INSTANCES.filter(c => c.status === "running").length} instances` },
    { label: "Queue Depth", value: WORKLOAD_QUEUE.filter(w => w.status === "queued").length, sub: `${WORKLOAD_QUEUE.filter(w => w.status === "running").length} running` },
    { label: "Cloud $/hr", value: `$${cloudHourlyCost.toFixed(0)}`, sub: "active instances" },
    { label: "Monthly Spend", value: `$${(monthlySpend / 1000).toFixed(1)}K`, sub: `of $${(monthlyBudget / 1000).toFixed(0)}K budget` },
    { label: "Spot Savings", value: "61%", sub: "vs on-demand" },
  ];

  return (
    <div className="space-y-5 animate-in fade-in duration-200">
      {/* KPI row */}
      <div className="grid grid-cols-6 gap-2.5">
        {kpis.map((m) => (
          <Card key={m.label} size="sm">
            <CardContent className="p-3">
              <div className="text-[10px] text-faint uppercase tracking-wider font-display mb-1">{m.label}</div>
              <div className="text-xl font-bold text-foreground tracking-tight">{m.value}</div>
              <div className="text-[11px] text-primary mt-0.5">{m.sub}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-[1fr_380px] gap-4">
        {/* Cluster map */}
        <Card size="sm">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
            <span className="text-sm font-semibold text-foreground font-display">Cluster Map</span>
            <div className="flex gap-2.5 text-[11px]">
              {[
                { label: "Training", color: WORKLOAD_COLORS.training },
                { label: "Serving", color: WORKLOAD_COLORS.serving },
                { label: "Eval", color: WORKLOAD_COLORS.eval },
                { label: "Idle", color: WORKLOAD_COLORS.idle },
              ].map(l => (
                <span key={l.label} className="flex items-center gap-1 text-faint">
                  <span className="w-2 h-2 rounded-sm" style={{ background: l.color }} />
                  {l.label}
                </span>
              ))}
            </div>
          </div>
          <div className="p-4 grid grid-cols-4 gap-2">
            {LOCAL_NODES.map(node => (
              <Card
                key={node.id}
                size="sm"
                className="p-2.5 cursor-pointer transition-all hover:border-border"
                style={{ opacity: node.status === "cordoned" ? 0.5 : 1 }}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[11px] text-muted-foreground font-medium">{node.id}</span>
                  <StatusDot status={node.status === "active" ? "running" : "error"} />
                </div>
                <div className="text-[10px] text-faint mb-1.5">
                  {node.gpu ? `${node.gpu.count}\u00d7 ${node.gpu.type}` : "CPU only"}
                </div>
                {node.gpu && (
                  <div className="grid gap-0.5 mb-1.5" style={{ gridTemplateColumns: `repeat(${node.gpu.count}, 1fr)` }}>
                    {node.workloads
                      .flatMap(w => Array.from({ length: w.gpu }, () => ({ color: WORKLOAD_COLORS[w.type] || WORKLOAD_COLORS.idle, label: w.name })))
                      .concat(Array.from({ length: Math.max(0, node.gpu.count - node.gpu.used) }, () => ({ color: WORKLOAD_COLORS.idle, label: "idle" })))
                      .slice(0, node.gpu.count)
                      .map((s, i) => (
                        <div key={i} className="h-4 rounded" style={{ background: s.color, border: `1px solid ${s.color === WORKLOAD_COLORS.idle ? "var(--border)" : s.color + "60"}` }} title={s.label} />
                      ))}
                  </div>
                )}
                <ProgressBar value={node.gpu?.utilization || node.cpu.utilization} color="var(--primary)" />
                <div className="text-[10px] text-faint mt-1">
                  GPU {node.gpu?.utilization || 0}% · CPU {node.cpu.utilization}% · {node.mem.used}/{node.mem.total}{node.mem.unit}
                </div>
              </Card>
            ))}
          </div>
        </Card>

        {/* Right sidebar */}
        <div className="flex flex-col gap-4">
          {/* Cloud instances */}
          <Card size="sm">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line">
              <span className="text-sm font-semibold text-foreground font-display">Cloud Instances</span>
              <span className="text-[11px] px-1.5 py-px rounded bg-muted text-faint">{CLOUD_INSTANCES.length}</span>
            </div>
            {CLOUD_INSTANCES.map(inst => (
              <div key={inst.id} className="px-4 py-2.5 border-b border-line last:border-0">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <StatusDot status={inst.status === "running" ? "running" : "deploying"} />
                    <span className="text-sm text-foreground font-medium">{inst.workload}</span>
                  </div>
                  <span className="text-sm font-semibold" style={{ color: "#F59E0B" }}>${inst.costPerHour}/hr</span>
                </div>
                <div className="flex gap-2 text-[11px] text-faint">
                  <span style={{ color: PROVIDER_COLORS[inst.provider] }}>{inst.provider.toUpperCase()}</span>
                  <span>{inst.region}</span>
                  <span>{inst.gpu}</span>
                  {inst.spotInstance && <span className="text-success">spot (\u2013{inst.spotSavings})</span>}
                </div>
                {inst.autoTerminate && inst.status === "running" && (
                  <div className="text-[11px] text-faint mt-0.5">
                    Auto-terminate in <span className="text-primary">{inst.autoTerminate}</span>
                  </div>
                )}
              </div>
            ))}
          </Card>

          {/* Budget gauge */}
          <Card size="sm" className="p-4">
            <div className="text-sm font-semibold text-foreground font-display mb-3">Monthly Budget</div>
            {CLOUD_ACCOUNTS.filter(a => a.status === "connected").map(a => (
              <div key={a.provider} className="mb-3">
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="flex items-center gap-1">
                    <span className="font-semibold" style={{ color: PROVIDER_COLORS[a.provider] }}>{a.provider.toUpperCase()}</span>
                    <span className="text-faint">{a.name}</span>
                  </span>
                  <span className="text-muted-foreground font-medium">${a.monthlySpend.toLocaleString()} / ${a.monthlyBudget.toLocaleString()}</span>
                </div>
                <ProgressBar value={(a.monthlySpend / a.monthlyBudget) * 100} color={PROVIDER_COLORS[a.provider]} />
              </div>
            ))}
          </Card>

          {/* Queue summary */}
          <Card size="sm" className="p-4">
            <div className="text-sm font-semibold text-foreground font-display mb-2.5">Queue Summary</div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Running", value: WORKLOAD_QUEUE.filter(w => w.status === "running").length, color: "#22C55E" },
                { label: "Queued", value: WORKLOAD_QUEUE.filter(w => w.status === "queued").length, color: "#6B7280" },
                { label: "Training", value: WORKLOAD_QUEUE.filter(w => w.type === "training").length, color: "#F59E0B" },
                { label: "Serving", value: WORKLOAD_QUEUE.filter(w => w.type === "serving").length, color: "#22C55E" },
              ].map(s => (
                <div key={s.label} className="flex justify-between py-1 text-[11px]">
                  <span className="text-faint">{s.label}</span>
                  <span className="font-semibold" style={{ color: s.color }}>{s.value}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
