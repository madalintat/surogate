// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";
import { Sparkline } from "@/components/ui/sparkline";

const SPARK_DATA = {
  agents: [12, 18, 14, 22, 19, 28, 25, 32, 29, 35, 31, 38, 42, 39, 45],
  latency: [320, 310, 340, 290, 300, 280, 310, 290, 270, 300, 280, 260, 270, 250, 260],
  tokens: [1.2, 1.4, 1.1, 1.6, 1.3, 1.8, 2.0, 1.7, 2.1, 2.4, 2.2, 2.6, 2.3, 2.8, 3.1],
  success: [89, 91, 88, 92, 90, 93, 91, 94, 92, 95, 93, 96, 94, 95, 97],
};

const METRIC_CARDS = [
  { label: "Active Agents", value: "4", sub: "+1 deploying", spark: SPARK_DATA.agents, color: "#22C55E" },
  { label: "Avg Response Time", value: "284ms", sub: "p99 across all", spark: SPARK_DATA.latency, color: "#3B82F6" },
  { label: "Tokens Today", value: "3.1M", sub: "\u2191 12% from yesterday", spark: SPARK_DATA.tokens, color: "#F59E0B" },
  { label: "Success Rate", value: "97.2%", sub: "Last 24 hours", spark: SPARK_DATA.success, color: "#8B5CF6" },
];

export function MetricCards() {
  return (
    <div className="grid grid-cols-4 gap-3 mb-5">
      {METRIC_CARDS.map((m) => (
        <Card key={m.label} className="px-4 py-3.5 flex justify-between items-end">
          <div>
            <div className="text-muted-foreground uppercase tracking-wider mb-1.5 font-display">
              {m.label}
            </div>
            <div className="text-[22px] font-bold text-foreground tracking-tight">
              {m.value}
            </div>
            <div className="mt-0.5" style={{ color: m.color }}>
              {m.sub}
            </div>
          </div>
          <Sparkline data={m.spark} color={m.color} />
        </Card>
      ))}
    </div>
  );
}
