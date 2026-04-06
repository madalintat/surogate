// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { PROVIDER_COLORS } from "../compute-data";

const INPUT =
  "w-full rounded-md border border-line bg-background px-3 py-2 text-sm text-foreground placeholder:text-faint focus:outline-none focus:ring-1 focus:ring-primary";

export function ConnectRunpod({ onCancel }: { onCancel: () => void }) {
  const [name, setName] = useState("");
  const [apiKey, setApiKey] = useState("");

  const color = PROVIDER_COLORS.runpod;
  const valid = name && apiKey;

  return (
    <>
      <div className="text-[12px] text-faint bg-surface rounded-md px-3 py-2 mb-5">
        Generate an API key from the{" "}
        <span className="text-foreground font-medium">RunPod dashboard &rarr; API Keys</span>.
        A single key provides full access to launch and manage GPU instances.
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Account Name</label>
          <input className={INPUT} placeholder="RunPod" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">API Key</label>
          <input type="password" className={INPUT} placeholder="Enter RunPod API key" value={apiKey} onChange={e => setApiKey(e.target.value)} />
        </div>
      </div>

      <div className="flex gap-3 mt-6 pt-4 border-t border-line">
        <Button size="sm" style={{ backgroundColor: color }} disabled={!valid}>Connect</Button>
        <Button variant="outline" size="sm" onClick={onCancel}>Cancel</Button>
      </div>
    </>
  );
}
