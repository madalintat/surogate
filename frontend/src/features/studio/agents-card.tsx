// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StatusDot } from "@/components/ui/status-dot";
import type { Status } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";

const AGENTS_DATA = [
  { name: "cx-support-v3", version: "3.2.1", status: "running" as Status, replicas: "3/3", cpu: "42%", mem: "1.8Gi", rps: 124, p99: "320ms", model: "llama-3.1-8b-cx" },
  { name: "code-assist-v2", version: "2.7.0", status: "running" as Status, replicas: "2/2", cpu: "67%", mem: "3.2Gi", rps: 89, p99: "890ms", model: "deepseek-r1-code" },
  { name: "data-analyst-v1", version: "1.4.0-rc2", status: "deploying" as Status, replicas: "1/2", cpu: "23%", mem: "0.9Gi", rps: 12, p99: "1.2s", model: "qwen-2.5-72b" },
  { name: "onboarding-bot", version: "1.0.3", status: "running" as Status, replicas: "1/1", cpu: "8%", mem: "0.4Gi", rps: 5, p99: "210ms", model: "llama-3.1-8b-cx" },
  { name: "safety-reviewer", version: "0.9.1", status: "error" as Status, replicas: "0/1", cpu: "0%", mem: "0Gi", rps: 0, p99: "\u2014", model: "guard-3b" },
];

const COLUMNS = ["Agent", "Version", "Status", "Replicas", "CPU", "MEM", "RPS", "P99", "Model"] as const;

export function AgentsCard() {
  return (
    <Card>
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
        <div className="flex items-center gap-2">
          <span style={{ color: "#F59E0B" }}>\u2B21</span>
          <span className="text-sm font-semibold text-foreground font-display">Deployed Agents</span>
          <Badge>5 agents</Badge>
        </div>
        <button
          type="button"
          className="bg-linear-to-br from-amber-500 to-amber-600 border-none rounded-[5px] px-3 py-[5px] font-semibold text-primary-foreground cursor-pointer font-display"
        >
          + Deploy
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-line">
              {COLUMNS.map((h) => (
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
                <td className="px-3 py-2.5 text-subtle">{a.replicas}</td>
                <td className="px-3 py-2.5 text-subtle">{a.cpu}</td>
                <td className="px-3 py-2.5 text-subtle">{a.mem}</td>
                <td className="px-3 py-2.5 text-foreground/80 font-medium">{a.rps}</td>
                <td className={cn("px-3 py-2.5", a.p99 === "\u2014" ? "text-faint" : "text-subtle")}>{a.p99}</td>
                <td className="px-3 py-2.5 text-muted-foreground">{a.model}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
