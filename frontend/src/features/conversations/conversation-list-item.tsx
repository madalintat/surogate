// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { cn } from "@/utils/cn";
import type { Conversation } from "./conversations-data";

export function ConversationListItem({
  convo,
  selected,
  onSelect,
}: {
  convo: Conversation;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={cn(
        "w-full text-left px-3.5 py-3 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-blue-500"
          : "border-l-transparent hover:bg-muted/30",
      )}
    >
      <div className="flex-1 min-w-0">
        {/* top line: model + service + time */}
        <div className="flex items-center gap-1.5 mb-0.5">
          <span className="text-[11px] font-semibold text-foreground font-display truncate">
            {convo.model || convo.runName}
          </span>
          <span className="text-[9px] text-muted-foreground/30">&middot;</span>
          <span className="text-[10px] text-muted-foreground truncate">
            {convo.projectName}
          </span>
          <span className="flex-1" />
          <span className="text-[9px] text-muted-foreground/30 shrink-0">
            {convo.time}
          </span>
        </div>

        {/* meta line */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[9px] text-muted-foreground/40">
            {convo.turnCount}t
          </span>
          <span className="text-[9px] text-muted-foreground/20">&middot;</span>
          <span className="text-[9px] text-muted-foreground/40">
            {(convo.tokensIn + convo.tokensOut).toLocaleString()} tok
          </span>
          {convo.avgLatencyMs != null && (
            <>
              <span className="text-[9px] text-muted-foreground/20">&middot;</span>
              <span className="text-[9px] text-muted-foreground/40">
                {Math.round(convo.avgLatencyMs)}ms avg
              </span>
            </>
          )}
          {convo.duration && (
            <>
              <span className="text-[9px] text-muted-foreground/20">&middot;</span>
              <span className="text-[9px] text-muted-foreground/40">
                {convo.duration}
              </span>
            </>
          )}
          {convo.hasCompaction && (
            <span
              className="text-[7px] px-1 py-px rounded font-semibold font-display"
              style={{ background: "#F59E0B12", color: "#F59E0B" }}
            >
              COMPACTED
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
