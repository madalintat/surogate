// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Commit } from "@/types/hub";

interface CommitsTabProps {
  commits: Commit[];
}

export function CommitsTab({ commits }: CommitsTabProps) {
  if (commits.length === 0) {
    return <div className="py-8 text-center text-faint">No commits in this ref</div>;
  }

  return (
    <div>
      {commits.map((c, i) => (
        <div key={c.id} className="flex items-start gap-3 py-3 border-b border-input">
          <div className="flex flex-col items-center w-4 shrink-0 pt-1">
            <div className="w-3 h-3 rounded-full border-2 bg-success border-success" />
            {i < commits.length - 1 && <div className="w-px flex-1 bg-border min-h-5 mt-1" />}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <code className="text-success font-medium">{c.id.slice(0, 8)}</code>
            </div>
            <div className="text-foreground/80 mb-0.5">{c.message}</div>
            <div className="text-faint">
              {c.committer} · {new Date(c.creation_date * 1000).toLocaleString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
