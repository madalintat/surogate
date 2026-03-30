// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { SCALING_POLICIES } from "./compute-data";

export function PoliciesTab() {
  return (
    <div className="animate-in fade-in duration-200">
      <div className="flex items-center justify-between mb-3">
        <span className="text-base font-bold text-foreground font-display">Auto-scaling & Resource Policies</span>
        <Button variant="outline" size="sm">+ New Policy</Button>
      </div>

      <div className="space-y-2.5">
        {SCALING_POLICIES.map(pol => (
          <Card
            key={pol.id}
            size="sm"
            className="p-4"
            style={{ opacity: pol.enabled ? 1 : 0.5 }}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2.5">
                <Switch checked={pol.enabled} size="sm" />
                <div>
                  <div className="text-sm font-semibold text-foreground font-display">{pol.name}</div>
                  <div className="text-[11px] text-faint mt-0.5">
                    Triggered {pol.triggerCount} times · last: {pol.lastTriggered}
                  </div>
                </div>
              </div>
              <Button variant="outline" size="xs">Edit</Button>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-background border border-line rounded-md p-2.5">
                <div className="text-[10px] text-primary uppercase tracking-wider font-display font-semibold mb-1">When</div>
                <div className="text-[11px] text-muted-foreground leading-relaxed">{pol.trigger}</div>
              </div>
              <div className="bg-background border border-line rounded-md p-2.5">
                <div className="text-[10px] text-success uppercase tracking-wider font-display font-semibold mb-1">Then</div>
                <div className="text-[11px] text-muted-foreground leading-relaxed">{pol.action}</div>
              </div>
            </div>

            {(pol.maxSpend !== "\u2014" || pol.cooldown !== "\u2014") && (
              <div className="flex gap-3 mt-2 text-[11px] text-faint">
                {pol.maxSpend !== "\u2014" && <span>max spend: <span className="text-primary">{pol.maxSpend}</span></span>}
                {pol.cooldown !== "\u2014" && <span>cooldown: <span className="text-muted-foreground">{pol.cooldown}</span></span>}
              </div>
            )}
          </Card>
        ))}
      </div>
    </div>
  );
}
