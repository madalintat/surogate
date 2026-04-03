// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { ProgressBar } from "@/components/ui/progress-bar";
import { CLOUD_ACCOUNTS, CLOUD_INSTANCES, PROVIDER_COLORS, STATUS_COLORS } from "./compute-data";

export function CloudTab() {
  const cloudHourlyCost = CLOUD_INSTANCES.filter(c => c.status === "running").reduce((s, c) => s + c.costPerHour, 0);

  return (
    <div className="space-y-5 animate-in fade-in duration-200">
      {/* Cloud accounts */}
      <div className="grid grid-cols-3 gap-3">
        {CLOUD_ACCOUNTS.map(a => (
          <Card
            key={a.provider}
            size="sm"
            className="p-4"
            style={{
              borderColor: a.status === "connected" ? (PROVIDER_COLORS[a.provider] ?? "#1E2330") + "30" : undefined,
              opacity: a.status === "disconnected" ? 0.4 : 1,
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-lg" style={{ color: PROVIDER_COLORS[a.provider] }}>{"\u2601"}</span>
                <div>
                  <div className="text-sm font-bold text-foreground font-display">{a.name}</div>
                  <div className="text-[11px] text-faint">{a.provider.toUpperCase()}</div>
                </div>
              </div>
              <StatusDot status={a.status === "connected" ? "running" : "stopped"} />
            </div>
            {a.status === "connected" && (
              <>
                <div className="grid grid-cols-2 gap-2 mb-2.5">
                  <div>
                    <div className="text-[10px] text-faint uppercase mb-0.5">GPU Quota</div>
                    <div className="text-base font-bold text-foreground">{a.usedGpu}/{a.quotaGpu}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-faint uppercase mb-0.5">Spend</div>
                    <div className="text-base font-bold text-foreground">${a.monthlySpend.toLocaleString()}</div>
                  </div>
                </div>
                <div className="mb-1">
                  <div className="flex justify-between text-[11px] text-faint mb-1">
                    <span>Budget: ${a.monthlyBudget.toLocaleString()}</span>
                    <span>{Math.round(a.monthlySpend / a.monthlyBudget * 100)}%</span>
                  </div>
                  <ProgressBar value={(a.monthlySpend / a.monthlyBudget) * 100} color={PROVIDER_COLORS[a.provider]} />
                </div>
                <div className="text-[11px] text-faint mt-2">Regions: {a.regions.join(", ")}</div>
              </>
            )}
            {a.status === "disconnected" && (
              <Button variant="outline" size="sm" className="w-full mt-2">Connect</Button>
            )}
          </Card>
        ))}
      </div>

      {/* Active Cloud instances */}
      <Card size="sm" className="overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <span className="text-sm font-semibold text-foreground font-display">Active Cloud Instances</span>
          <span className="text-sm" style={{ color: "#F59E0B" }}>Total: ${cloudHourlyCost.toFixed(2)}/hr</span>
        </div>
        {CLOUD_INSTANCES.map(inst => (
          <div key={inst.id} className="px-4 py-3.5 border-b border-line last:border-0 flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <StatusDot status={inst.status === "running" ? "running" : "deploying"} />
                <span className="text-sm font-semibold text-foreground">{inst.workload}</span>
                {inst.status === "provisioning" && (
                  <span className="text-[10px] px-1.5 py-px rounded bg-[#06B6D412] text-[#06B6D4] animate-pulse">PROVISIONING</span>
                )}
              </div>
              <div className="flex gap-3 text-[11px] text-faint">
                <span style={{ color: PROVIDER_COLORS[inst.provider] }}>{inst.provider.toUpperCase()} · {inst.region}</span>
                <span>{inst.type}</span>
                <span>{inst.gpu}</span>
                {inst.spotInstance && <span className="text-success">Spot (\u2013{inst.spotSavings})</span>}
                <span>started {inst.startedAt}</span>
              </div>
            </div>
            <div className="text-right shrink-0">
              <div className="text-lg font-bold" style={{ color: "#F59E0B" }}>${inst.costPerHour}/hr</div>
              {inst.estimatedTotal > 0 && <div className="text-[11px] text-faint">est: ${inst.estimatedTotal.toFixed(0)} total</div>}
              {inst.autoTerminate && <div className="text-[11px] text-destructive mt-0.5">auto-terminate: {inst.autoTerminate}</div>}
              <Button variant="outline" size="xs" className="mt-1.5 text-destructive border-destructive/30">Terminate</Button>
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
}
