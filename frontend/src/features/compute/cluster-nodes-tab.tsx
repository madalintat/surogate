// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StatusDot } from "@/components/ui/status-dot";
import { ProgressBar } from "@/components/ui/progress-bar";
import { LOCAL_NODES, WORKLOAD_COLORS, STATUS_COLORS } from "./compute-data";

const POOL_STYLE: Record<string, { bg: string; text: string }> = {
  training: { bg: "bg-[#F59E0B12]", text: "text-[#F59E0B]" },
  serving: { bg: "bg-[#22C55E12]", text: "text-[#22C55E]" },
  system: { bg: "bg-muted", text: "text-faint" },
};

export function ClusterNodesTab() {
  return (
    <div className="animate-in fade-in duration-200">
      <Card size="sm" className="overflow-hidden">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-line">
              {["Node", "Pool", "Status", "GPU", "GPU Util", "CPU", "Memory", "Workloads"].map(h => (
                <th key={h} className="px-3.5 py-2 text-left text-[10px] font-medium text-faint uppercase tracking-wider font-display">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {LOCAL_NODES.map(n => {
              const pool = POOL_STYLE[n.pool] ?? POOL_STYLE.system;
              return (
                <tr
                  key={n.id}
                  className="border-b border-line last:border-0 hover:bg-muted/50 transition-colors"
                  style={{ opacity: n.status === "cordoned" ? 0.5 : 1 }}
                >
                  <td className="px-3.5 py-2.5">
                    <div className="text-sm text-foreground font-medium">{n.id}</div>
                    <div className="text-[11px] text-faint">{n.hostname}</div>
                  </td>
                  <td className="px-3.5 py-2.5">
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium uppercase ${pool.bg} ${pool.text}`}>{n.pool}</span>
                  </td>
                  <td className="px-3.5 py-2.5">
                    <span className="flex items-center gap-1 text-[11px]">
                      <StatusDot status={n.status === "active" ? "running" : n.status === "cordoned" ? "stopped" : "error"} />
                      <span style={{ color: STATUS_COLORS[n.status] }}>{n.status}</span>
                    </span>
                  </td>
                  <td className="px-3.5 py-2.5 text-[11px] text-muted-foreground">
                    {n.gpu ? `${n.gpu.used}/${n.gpu.count} ${n.gpu.type}` : "\u2014"}
                  </td>
                  <td className="px-3.5 py-2.5 w-24">
                    {n.gpu ? <ProgressBar value={n.gpu.utilization} color="var(--primary)" /> : <span className="text-[11px] text-faint">\u2014</span>}
                  </td>
                  <td className="px-3.5 py-2.5 text-[11px] text-muted-foreground">
                    {n.cpu.used}/{n.cpu.cores} cores ({n.cpu.utilization}%)
                  </td>
                  <td className="px-3.5 py-2.5 text-[11px] text-muted-foreground">
                    {n.mem.used}/{n.mem.total} {n.mem.unit}
                  </td>
                  <td className="px-3.5 py-2.5">
                    <div className="flex gap-1 flex-wrap">
                      {n.workloads.filter(w => w.type !== "idle").map((w, i) => (
                        <span
                          key={i}
                          className="text-[10px] px-1.5 py-px rounded border"
                          style={{
                            background: (WORKLOAD_COLORS[w.type] ?? "#3A4154") + "15",
                            color: WORKLOAD_COLORS[w.type] ?? "#3A4154",
                            borderColor: (WORKLOAD_COLORS[w.type] ?? "#3A4154") + "25",
                          }}
                        >
                          {w.name}
                        </span>
                      ))}
                      {n.workloads.length === 0 && <span className="text-[11px] text-faint">idle</span>}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
