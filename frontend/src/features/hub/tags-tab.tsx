// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Tag } from "lucide-react";
import { Card } from "@/components/ui/card";
import type { Ref } from "@/types/hub";

interface TagsTabProps {
  tags: Ref[];
  onSelectRef: (ref: string) => void;
}

export function TagsTab({ tags, onSelectRef }: TagsTabProps) {
  if (tags.length === 0) {
    return <div className="py-8 text-center text-faint">No tags</div>;
  }

  return (
    <Card>
      {tags.map((t) => (
        <div
          key={t.id}
          className="px-4 py-2.5 border-b border-input flex items-center gap-2 cursor-pointer hover:bg-input transition-colors"
          onClick={() => onSelectRef(t.id)}
        >
          <Tag size={14} className="text-muted-foreground" />
          <span className="text-foreground">{t.id}</span>
          <code className="text-faint ml-auto">{t.commit_id.slice(0, 8)}</code>
        </div>
      ))}
    </Card>
  );
}
