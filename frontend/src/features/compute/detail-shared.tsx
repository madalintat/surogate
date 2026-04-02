// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { WorkloadItem } from "./compute-data";
import type { LocalTask } from "@/api/tasks";
import type { Model } from "@/types/model";

export type ExtendedWorkload = WorkloadItem & { _task?: LocalTask; _model?: Model };

export const TASK_COLOR = "#06B6D4";

export const EXTENDED_WORKLOAD_COLORS: Record<string, string> = {
  training: "#F59E0B",
  serving: "#22C55E",
  eval: "#8B5CF6",
  system: "#3A4154",
  idle: "#1A1F2E",
  task: TASK_COLOR,
};

export function statusForDot(s: string) {
  if (s === "running" || s === "serving" || s === "ready") return "running" as const;
  if (s === "provisioning" || s === "pending" || s === "deploying"
    || s === "controller_init" || s === "replica_init" || s === "no_replica") return "deploying" as const;
  if (s === "completed") return "completed" as const;
  if (s === "failed" || s === "error" || s === "controller_failed" || s === "failed_cleanup") return "error" as const;
  return "stopped" as const;
}

export function parseProgress(progress: string): number {
  const match = progress.match(/([\d.]+)%/);
  return match ? parseFloat(match[1]) : 0;
}

export function InfoRow({ icon: Icon, label, value }: {
  icon: React.ComponentType<{ size?: number; className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-start gap-1.5">
      <Icon size={11} className="text-faint mt-0.5 shrink-0" />
      <div className="min-w-0">
        <div className="text-[10px] text-faint uppercase tracking-wider">{label}</div>
        <div className="text-[11px] text-foreground truncate">{value}</div>
      </div>
    </div>
  );
}
