// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { ConversationDetail } from "@/types/conversation";

export function TrajectoryTab({ convo }: { convo: ConversationDetail }) {
  return (
    <div className="animate-in fade-in duration-150">
      <div className="text-xs font-semibold text-foreground font-display mb-3">
        Conversation Trajectory
      </div>

      {/* timeline */}
      <div className="relative pl-7">
        {convo.messages.map((msg, i) => {
          const isUser = msg.role === "user";
          const isTool = msg.role === "tool";
          const dotColor = isTool
            ? "#F59E0B"
            : isUser
              ? "var(--color-muted-foreground)"
              : "#3B82F6";

          return (
            <div key={i} className="relative mb-1">
              {/* vertical line */}
              <div className="absolute -left-5 top-0 bottom-0 w-px bg-border" />
              {/* dot */}
              <div
                className="absolute -left-6 top-1.5 w-2 h-2 rounded-full border-2"
                style={{ background: dotColor, borderColor: dotColor }}
              />

              <div className="px-3 py-1.5 rounded-md">
                {/* header */}
                <div className="flex items-center gap-1.5 text-[10px]">
                  <span
                    className="font-medium font-display"
                    style={{ color: dotColor }}
                  >
                    {isTool ? "Tool" : isUser ? "User" : "Agent"}
                  </span>
                  <span className="text-muted-foreground/30 text-[9px]">
                    {msg.timestamp}
                  </span>
                  <span className="text-muted-foreground/30 text-[9px]">
                    {msg.tokens} tok
                  </span>
                  {msg.latency && (
                    <span className="text-muted-foreground/30 text-[9px]">
                      {msg.latency}ms
                    </span>
                  )}
                </div>

                {/* content preview */}
                <div className="text-[10px] text-muted-foreground mt-0.5 truncate max-w-[500px]">
                  {msg.content
                    ? <>
                        {msg.content.substring(0, 120)}
                        {msg.content.length > 120 ? "..." : ""}
                      </>
                    : <span className="text-muted-foreground/30 italic">__EMPTY__</span>
                  }
                </div>
              </div>
            </div>
          );
        })}

        {convo.messages.length === 0 && (
          <div className="text-[11px] text-muted-foreground/30 py-4">
            No messages recorded
          </div>
        )}
      </div>

      {/* turn stats */}
      <div className="mt-5 grid grid-cols-3 gap-2.5">
        <div className="bg-muted/40 border border-border rounded-lg p-3">
          <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">
            Total Turns
          </div>
          <div className="text-xl font-bold text-foreground">
            {convo.turnCount}
          </div>
        </div>

        <div className="bg-muted/40 border border-border rounded-lg p-3">
          <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">
            Avg Latency
          </div>
          <div className="text-xl font-bold text-blue-500">
            {convo.avgLatencyMs != null ? `${Math.round(convo.avgLatencyMs)}ms` : "--"}
          </div>
        </div>

        <div className="bg-muted/40 border border-border rounded-lg p-3">
          <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">
            Compactions
          </div>
          <div className="text-xl font-bold text-amber-500">
            {convo.turns.filter((t) => t.compacted).length}
          </div>
          <div className="text-[9px] text-muted-foreground/30 mt-0.5">
            {convo.hasCompaction ? "History was compacted" : "No compaction detected"}
          </div>
        </div>
      </div>
    </div>
  );
}
