// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { CLOUD_ACCOUNTS, COST_DAILY, COST_BY_TYPE, COST_BY_PROJECT } from "./compute-data";

export function CostsTab() {
  const monthlySpend = CLOUD_ACCOUNTS.reduce((s, a) => s + a.monthlySpend, 0);
  const maxDaily = Math.max(...COST_DAILY.map(d => d.value));

  return (
    <div className="space-y-4 animate-in fade-in duration-200">
      {/* Daily spend chart */}
      <Card size="sm" className="overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <span className="text-sm font-semibold text-foreground font-display">Daily Cloud Spend (30 days)</span>
          <span className="text-base font-bold" style={{ color: "#F59E0B" }}>
            ${monthlySpend.toLocaleString()}{" "}
            <span className="text-[11px] font-normal text-faint">this month</span>
          </span>
        </div>
        <div className="px-4 pt-4 pb-2">
          <div className="flex items-end gap-px h-20">
            {COST_DAILY.map((d, i) => (
              <div
                key={i}
                className="flex-1 rounded-sm transition-all"
                style={{
                  height: `${(d.value / maxDaily) * 70}px`,
                  background: i === COST_DAILY.length - 1 ? "#F59E0B" : "var(--primary)",
                  opacity: i === COST_DAILY.length - 1 ? 1 : 0.3,
                }}
                title={`Day ${d.day}: $${d.value}`}
              />
            ))}
          </div>
          <div className="flex justify-between text-[11px] text-faint mt-1.5 font-display">
            <span>30 days ago</span>
            <span>today</span>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        {/* By type */}
        <Card size="sm" className="p-4">
          <div className="text-sm font-semibold text-foreground font-display mb-3.5">By Job Type</div>
          {COST_BY_TYPE.map(c => (
            <div key={c.type} className="mb-2.5">
              <div className="flex justify-between text-[11px] mb-1">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-sm" style={{ background: c.color }} />
                  <span className="text-muted-foreground font-display">{c.type}</span>
                </span>
                <span className="text-foreground font-semibold">${c.cost.toLocaleString()}</span>
              </div>
              <ProgressBar value={c.pct} color={c.color} />
            </div>
          ))}
        </Card>

        {/* By project */}
        <Card size="sm" className="p-4">
          <div className="text-sm font-semibold text-foreground font-display mb-3.5">By Project</div>
          {COST_BY_PROJECT.map(c => (
            <div key={c.project} className="mb-2.5">
              <div className="flex justify-between text-[11px] mb-1">
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-sm" style={{ background: c.color }} />
                  <span className="text-muted-foreground font-display">{c.project}</span>
                </span>
                <span className="text-foreground font-semibold">${c.cost.toLocaleString()}</span>
              </div>
              <ProgressBar value={c.pct} color={c.color} />
            </div>
          ))}
          <div className="mt-3 pt-3 border-t border-line flex justify-between text-sm">
            <span className="text-muted-foreground font-display">Total</span>
            <span className="text-foreground font-bold">${COST_BY_PROJECT.reduce((s, c) => s + c.cost, 0).toLocaleString()}</span>
          </div>
        </Card>
      </div>
    </div>
  );
}
