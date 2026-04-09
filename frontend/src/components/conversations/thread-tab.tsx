// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { ConversationDetail, ConversationMessage } from "@/types/conversation";

function MessageBubble({ msg }: { msg: ConversationMessage }) {
  const isUser = msg.role === "user";
  const isTool = msg.role === "tool";

  if (isTool) {
    return (
      <div className="flex gap-2.5">
        <div
          className="w-7 h-7 rounded-md shrink-0 flex items-center justify-center text-[10px] font-semibold font-display border"
          style={{ backgroundColor: "#F59E0B18", borderColor: "#F59E0B30", color: "#F59E0B" }}
        >
          T
        </div>
        <div
          className="max-w-[75%] border px-3.5 py-2.5 relative"
          style={{
            background: "var(--color-card)",
            borderColor: "var(--color-border)",
            borderRadius: "10px 10px 10px 4px",
          }}
        >
          {msg.toolCallId && (
            <div className="text-[9px] text-muted-foreground/40 font-mono mb-1">
              {msg.toolCallId}
            </div>
          )}
          <div className="text-[12px] text-foreground leading-relaxed whitespace-pre-wrap break-words font-mono">
            {msg.content || <span className="text-muted-foreground/30 italic">empty</span>}
          </div>
          <div className="flex items-center mt-1.5 text-[9px] text-muted-foreground/30">
            <span>{msg.timestamp}</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex gap-2.5"
      style={{ flexDirection: isUser ? "row-reverse" : "row" }}
    >
      {/* avatar */}
      <div
        className="w-7 h-7 rounded-md shrink-0 flex items-center justify-center text-[10px] font-semibold font-display border"
        style={{
          backgroundColor: isUser ? undefined : "#3B82F618",
          borderColor: isUser ? undefined : "#3B82F630",
          color: isUser ? undefined : "#3B82F6",
        }}
      >
        {isUser ? "U" : "A"}
      </div>

      {/* bubble */}
      <div
        className="max-w-[75%] border px-3.5 py-2.5 relative"
        style={{
          background: isUser
            ? "var(--color-muted)"
            : "var(--color-card)",
          borderColor: "var(--color-border)",
          borderRadius: isUser
            ? "10px 10px 4px 10px"
            : "10px 10px 10px 4px",
        }}
      >
        {/* reasoning (thinking) */}
        {msg.reasoningContent && (
          <details className="mb-1.5">
            <summary className="text-[9px] text-muted-foreground/40 cursor-pointer select-none font-display">
              Thinking
            </summary>
            <div className="text-[11px] text-muted-foreground/50 leading-relaxed whitespace-pre-wrap mt-1 pl-2 border-l border-border/50">
              {msg.reasoningContent}
            </div>
          </details>
        )}

        {/* content */}
        {(msg.content || !msg.toolCalls) && (
          <div className="text-[12px] text-foreground leading-relaxed whitespace-pre-wrap break-words">
            {msg.content || <span className="text-muted-foreground/30 italic">__EMPTY__</span>}
          </div>
        )}

        {/* tool calls */}
        {msg.toolCalls && msg.toolCalls.length > 0 && (
          <div className="space-y-1.5 mt-1">
            {msg.toolCalls.map((tc, j) => (
              <div
                key={tc.id || j}
                className="border border-amber-500/20 bg-amber-500/5 rounded px-2.5 py-1.5"
              >
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className="text-[9px] font-semibold text-amber-500 font-display uppercase">
                    Tool Call
                  </span>
                  <span className="text-[10px] font-medium text-foreground font-mono">
                    {tc.function.name}
                  </span>
                </div>
                {tc.function.arguments && (
                  <pre className="text-[10px] text-muted-foreground leading-relaxed overflow-x-auto whitespace-pre-wrap break-all font-mono">
                    {tc.function.arguments}
                  </pre>
                )}
              </div>
            ))}
          </div>
        )}

        {/* meta footer */}
        <div className="flex items-center justify-between mt-1.5 text-[9px] text-muted-foreground/30">
          <span>{msg.timestamp}</span>
          <div className="flex gap-2">
            {msg.tokens != null && <span>{msg.tokens} tok</span>}
            {msg.latency && <span>{msg.latency}ms</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

export function ThreadTab({ convo }: { convo: ConversationDetail }) {
  return (
    <div className="animate-in fade-in duration-150 space-y-3">
      {convo.messages.map((msg, i) => (
        <MessageBubble key={i} msg={msg} />
      ))}

      {convo.messages.length === 0 && (
        <div className="text-center text-muted-foreground/30 text-xs py-8">
          No messages recorded for this conversation
        </div>
      )}
    </div>
  );
}
