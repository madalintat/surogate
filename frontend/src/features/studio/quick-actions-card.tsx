// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";

const QUICK_ACTIONS = [
  { icon: "\u2B21", label: "Deploy Agent", desc: "From template or config" },
  { icon: "\u25C7", label: "Serve Model", desc: "Deploy LLM to cluster" },
  { icon: "\u26A1", label: "New Skill", desc: "Create reusable skill" },
  { icon: "\u25EC", label: "Start Training", desc: "SFT or RL job" },
  { icon: "\u25C8", label: "Run Eval", desc: "Benchmark or custom" },
  { icon: "\u25A4", label: "Import Dataset", desc: "From conversations" },
];

export function QuickActionsCard() {
  return (
    <Card>
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line">
        <span style={{ color: "#F59E0B" }}>\u25EB</span>
        <span className="text-sm font-semibold text-foreground font-display">Quick Actions</span>
      </div>
      <div className="p-2 grid grid-cols-2 gap-1.5">
        {QUICK_ACTIONS.map((a) => (
          <button
            key={a.label}
            type="button"
            className="flex flex-col items-start px-3 py-2.5 rounded-md border border-transparent bg-transparent cursor-pointer text-left transition-all duration-150 hover:border-border hover:bg-input"
          >
            <span className="text-base mb-1">{a.icon}</span>
            <span className="text-foreground font-medium font-display">{a.label}</span>
            <span className="text-[9px] text-faint">{a.desc}</span>
          </button>
        ))}
      </div>
    </Card>
  );
}
