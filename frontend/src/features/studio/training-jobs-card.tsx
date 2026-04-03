// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusDot } from "@/components/ui/status-dot";
import type { Status } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";

const TRAINING_JOBS = [
  { id: "ft-0042", name: "CX SFT Round 4", type: "SFT", status: "running" as Status, progress: 67, epoch: "2/3", loss: "0.847", compute: "local", gpu: "4\u00d7 H100" },
  { id: "ft-0041", name: "Code RL Phase 2", type: "GRPO", status: "running" as Status, progress: 34, epoch: "1/5", loss: "1.203", compute: "aws", gpu: "8\u00d7 A100" },
  { id: "ft-0040", name: "Guard classifier v2", type: "SFT", status: "completed" as Status, progress: 100, epoch: "3/3", loss: "0.312", compute: "local", gpu: "1\u00d7 A100" },
];

export function TrainingJobsCard() {
  return (
    <Card>
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line">
        <span style={{ color: "#8B5CF6" }}>\u25EC</span>
        <span className="text-sm font-semibold text-foreground font-display">Training Jobs</span>
      </div>
      {TRAINING_JOBS.map((j) => (
        <div key={j.id} className="px-4 py-2.5 border-b border-input">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <StatusDot status={j.status} />
              <span className="text-foreground font-medium">{j.name}</span>
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
              {j.compute === "aws" ? "\u2601 AWS" : "\u229e Local"}
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
  );
}
