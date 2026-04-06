// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { PROVIDER_COLORS } from "../compute-data";

const INPUT =
  "w-full rounded-md border border-line bg-background px-3 py-2 text-sm text-foreground placeholder:text-faint focus:outline-none focus:ring-1 focus:ring-primary";

export function ConnectAws({ onCancel }: { onCancel: () => void }) {
  const [name, setName] = useState("");
  const [accessKeyId, setAccessKeyId] = useState("");
  const [secretAccessKey, setSecretAccessKey] = useState("");
  const [region, setRegion] = useState("");
  const [roleArn, setRoleArn] = useState("");

  const color = PROVIDER_COLORS.aws;
  const valid = name && accessKeyId && secretAccessKey && region;

  return (
    <>
      <div className="text-[12px] text-faint bg-surface rounded-md px-3 py-2 mb-5">
        Create an IAM user with <span className="text-foreground font-medium">AmazonEC2FullAccess</span> and{" "}
        <span className="text-foreground font-medium">AmazonS3ReadOnlyAccess</span> policies.
        For cross-account setups, provide a Role ARN to assume.
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Account Name</label>
          <input className={INPUT} placeholder="AWS Production" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Default Region</label>
          <input className={INPUT} placeholder="us-east-1" value={region} onChange={e => setRegion(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Access Key ID</label>
          <input className={INPUT} placeholder="AKIA..." value={accessKeyId} onChange={e => setAccessKeyId(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Secret Access Key</label>
          <input type="password" className={INPUT} placeholder="Enter secret access key" value={secretAccessKey} onChange={e => setSecretAccessKey(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Role ARN <span className="normal-case text-faint/60">(optional, for cross-account)</span></label>
          <input className={INPUT} placeholder="arn:aws:iam::123456789012:role/SurogateRole" value={roleArn} onChange={e => setRoleArn(e.target.value)} />
        </div>
      </div>

      <div className="flex gap-3 mt-6 pt-4 border-t border-line">
        <Button size="sm" style={{ backgroundColor: color }} disabled={!valid}>Connect</Button>
        <Button variant="outline" size="sm" onClick={onCancel}>Cancel</Button>
      </div>
    </>
  );
}
