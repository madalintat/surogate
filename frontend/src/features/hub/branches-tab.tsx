// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { GitBranch } from "lucide-react";
import { Card } from "@/components/ui/card";
import { cn } from "@/utils/cn";
import type { Ref } from "@/types/hub";

interface BranchesTabProps {
  branches: Ref[];
  activeRef: string;
  onSelectRef: (ref: string) => void;
}

export function BranchesTab({ branches, activeRef, onSelectRef }: BranchesTabProps) {
  if (branches.length === 0) {
    return <div className="py-8 text-center text-faint">No branches</div>;
  }

  return (
    <Card>
      {branches.map((b) => (
        <div
          key={b.id}
          className="px-4 py-2.5 border-b border-input flex items-center gap-2 cursor-pointer hover:bg-input transition-colors"
          onClick={() => onSelectRef(b.id)}
        >
          <GitBranch size={14} className="text-muted-foreground" />
          <span className={cn("text-foreground", b.id === activeRef && "font-semibold text-success")}>{b.id}</span>
          <code className="text-faint ml-auto">{b.commit_id.slice(0, 8)}</code>
        </div>
      ))}
    </Card>
  );
}
