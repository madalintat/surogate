// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";
import { PROVIDER_COLORS } from "../compute-data";

const INPUT =
  "w-full rounded-md border border-line bg-background px-3 py-2 text-sm text-foreground placeholder:text-faint focus:outline-none focus:ring-1 focus:ring-primary";

export function ConnectGcp({ onCancel }: { onCancel: () => void }) {
  const [name, setName] = useState("");
  const [projectId, setProjectId] = useState("");
  const [keyFileName, setKeyFileName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const color = PROVIDER_COLORS.gcp;
  const valid = name && projectId && keyFileName;

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) setKeyFileName(file.name);
  }

  return (
    <>
      <div className="text-[12px] text-faint bg-surface rounded-md px-3 py-2 mb-5">
        Create a Service Account in your GCP project with <span className="text-foreground font-medium">Compute Admin</span> and{" "}
        <span className="text-foreground font-medium">Storage Object Viewer</span> roles, then download a JSON key file.
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Account Name</label>
          <input className={INPUT} placeholder="GCP Research" value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Project ID</label>
          <input className={INPUT} placeholder="my-project-123456" value={projectId} onChange={e => setProjectId(e.target.value)} />
        </div>
        <div>
          <label className="block text-xs font-medium text-faint uppercase mb-1.5">Service Account Key (JSON)</label>
          <input ref={fileRef} type="file" accept=".json" className="hidden" onChange={handleFile} />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className={`${INPUT} flex items-center gap-2 cursor-pointer text-left`}
          >
            <Upload size={14} className="shrink-0 text-faint" />
            <span className={keyFileName ? "text-foreground" : "text-faint"}>
              {keyFileName || "Upload service account JSON key"}
            </span>
          </button>
        </div>
      </div>

      <div className="flex gap-3 mt-6 pt-4 border-t border-line">
        <Button size="sm" style={{ backgroundColor: color }} disabled={!valid}>Connect</Button>
        <Button variant="outline" size="sm" onClick={onCancel}>Cancel</Button>
      </div>
    </>
  );
}
