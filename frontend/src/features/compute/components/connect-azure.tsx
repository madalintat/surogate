// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { PROVIDER_COLORS } from "../compute-data";

const INPUT =
  "w-full rounded-md border border-line bg-background px-3 py-2 text-sm text-foreground placeholder:text-faint focus:outline-none focus:ring-1 focus:ring-primary";

export function ConnectAzure({ onCancel }: { onCancel: () => void }) {
  const [name, setName] = useState("");
  const [tenantId, setTenantId] = useState("");
  const [subscriptionId, setSubscriptionId] = useState("");
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");

  const color = PROVIDER_COLORS.azure;
  const valid = name && tenantId && subscriptionId && clientId && clientSecret;

  return (
    <>
      <div className="text-[12px] text-faint bg-surface rounded-md px-3 py-2 mb-5">
        Register an App in Azure Active Directory and assign it the{" "}
        <span className="text-foreground font-medium">Contributor</span> role on the target subscription.
        You'll need the Tenant ID, Subscription ID, and the app's Client ID / Secret.
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Account Name</label>
          <input className={INPUT} placeholder="Azure Production" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-faint uppercase mb-1.5">Tenant ID</label>
            <input className={INPUT} placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" value={tenantId} onChange={e => setTenantId(e.target.value)} />
          </div>
          <div>
            <label className="block text-xs font-medium text-faint uppercase mb-1.5">Subscription ID</label>
            <input className={INPUT} placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" value={subscriptionId} onChange={e => setSubscriptionId(e.target.value)} />
          </div>
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Client ID (App ID)</label>
          <input className={INPUT} placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" value={clientId} onChange={e => setClientId(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Client Secret</label>
          <input type="password" className={INPUT} placeholder="Enter client secret" value={clientSecret} onChange={e => setClientSecret(e.target.value)} />
        </div>
      </div>

      <div className="flex gap-3 mt-6 pt-4 border-t border-line">
        <Button size="sm" style={{ backgroundColor: color }} disabled={!valid}>Connect</Button>
        <Button variant="outline" size="sm" onClick={onCancel}>Cancel</Button>
      </div>
    </>
  );
}
