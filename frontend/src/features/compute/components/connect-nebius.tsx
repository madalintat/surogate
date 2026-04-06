// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export function ConnectNebius({ onCancel }: { onCancel: () => void }) {
  const [name, setName] = useState("");
  const [apiKey, setApiKey] = useState("");

  const valid = name && apiKey;

  return (
    <>
      <div className="text-[12px] text-faint bg-surface rounded-md pb-2">
        Generate an API key from the{" "}
        <span className="text-foreground font-medium">Lambda Cloud dashboard &rarr; API Keys</span>.
        A single key provides full access to launch and manage GPU instances.
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5" htmlFor="account-name">Account Name</label>
          <Input id="account-name" className="h-8 text-xs" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5" htmlFor="api-key">API Key</label>
          <Input id="api-key" type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} />
        </div>
      </div>

      <div className="flex gap-3 mt-6 pt-4 border-t border-line">
        <Button size="sm" disabled={!valid}>Connect</Button>
        <Button variant="outline" size="sm" onClick={onCancel}>Cancel</Button>
      </div>
    </>
  );
}
