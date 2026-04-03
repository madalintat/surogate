// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";

const ACTIVITY = [
  { time: "2m", icon: "\u2B21", text: "data-analyst-v1 scaling to 2 replicas", color: "#F59E0B" },
  { time: "8m", icon: "\u25EC", text: "CX SFT Round 4 \u2014 epoch 2 started", color: "#8B5CF6" },
  { time: "14m", icon: "\u25C8", text: "GSM8K eval completed: 82.4%", color: "#3B82F6" },
  { time: "22m", icon: "\u2295", text: "llama-3.1-8b-cx-v4 pushed to Hub", color: "#22C55E" },
  { time: "31m", icon: "\u26A1", text: "New skill order-lookup added to CX", color: "#F59E0B" },
  { time: "45m", icon: "\u22A1", text: "2 conversations flagged for review", color: "#EF4444" },
  { time: "1h", icon: "\u25C7", text: "guard-3b OOM \u2014 restarting with 16Gi", color: "#EF4444" },
];

export function ActivityCard() {
  return (
    <Card>
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line">
        <span style={{ color: "var(--subtle)" }}>\u2299</span>
        <span className="text-sm font-semibold text-foreground font-display">Activity</span>
      </div>
      <div className="py-1">
        {ACTIVITY.map((e) => (
          <div
            key={e.time + e.text}
            className="px-4 py-[7px] flex items-start gap-2.5 cursor-pointer transition-colors duration-100 hover:bg-input"
          >
            <span className="text-faint text-[9px] w-6 text-right shrink-0 mt-0.5 font-display">
              {e.time}
            </span>
            <span className="shrink-0" style={{ color: e.color }}>
              {e.icon}
            </span>
            <span className="text-subtle">{e.text}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
