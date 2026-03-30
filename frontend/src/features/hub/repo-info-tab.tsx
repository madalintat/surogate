// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { getObjectContent } from "@/api/hub";
import type { Repository } from "@/types/hub";

interface RepoInfoTabProps {
  repo: Repository;
  activeRef: string;
  typeLabel: string;
  tags: string[];
}

export function RepoInfoTab({ repo, activeRef, typeLabel, tags }: RepoInfoTabProps) {
  const [readme, setReadme] = useState<string | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    setReadme(undefined); // eslint-disable-line react-hooks/set-state-in-effect -- reset before async fetch
    getObjectContent(repo.id, activeRef, "README.md")
      .then((content) => { if (!cancelled) setReadme(content); })
      .catch(() => { if (!cancelled) setReadme(null); });
    return () => { cancelled = true; };
  }, [repo.id, activeRef]);

  return (
    <div>
      {/* info bar */}
      <div className="flex items-center gap-3 flex-wrap mb-5 text-faint">
        <span><span className="text-muted-foreground">Repository ID:</span> <span className="text-foreground">{repo.id}</span></span>
        <span>·</span>
        <span><span className="text-muted-foreground">Type:</span> <span className="text-foreground">{typeLabel}</span></span>
        <span>·</span>
        <span><span className="text-muted-foreground">Created:</span> <span className="text-foreground">{new Date(repo.creation_date * 1000).toLocaleDateString()}</span></span>
        {tags.length > 0 && (
          <>
            <span>·</span>
            <div className="flex items-center gap-1">
              {tags.map((t) => (
                <span key={t} className="px-1.5 py-px rounded bg-accent text-muted-foreground">{t}</span>
              ))}
            </div>
          </>
        )}
      </div>

      {/* README.md */}
      {readme === undefined && (
        <div className="space-y-3">
          <Skeleton className="h-6 w-3/4" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}
      {readme !== undefined && readme !== null && (
        <MarkdownPreview
          markdown={readme}
          className="!max-h-none !text-base !p-5 !leading-normal"
        />
      )}
      {readme === null && (
        <div className="py-8 text-center text-faint">No README.md found in this repository</div>
      )}
    </div>
  );
}
