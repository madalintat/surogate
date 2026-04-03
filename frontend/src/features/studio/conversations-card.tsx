// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/utils/cn";

const CONVERSATIONS_RECENT = [
  { id: "c-9821", agent: "cx-support-v3", user: "user_8472", preview: "I need to change my subscription plan to...", tokens: 2340, turns: 8, sentiment: "positive", flagged: false, time: "2m ago" },
  { id: "c-9820", agent: "cx-support-v3", user: "user_3109", preview: "Why was I charged twice for the same...", tokens: 4120, turns: 14, sentiment: "negative", flagged: true, time: "5m ago" },
  { id: "c-9819", agent: "code-assist-v2", user: "dev_riley", preview: "Help me refactor this React component to use...", tokens: 8900, turns: 22, sentiment: "positive", flagged: false, time: "8m ago" },
  { id: "c-9818", agent: "data-analyst-v1", user: "analyst_jen", preview: "Generate a quarterly revenue breakdown by...", tokens: 3200, turns: 6, sentiment: "neutral", flagged: false, time: "12m ago" },
  { id: "c-9817", agent: "cx-support-v3", user: "user_0091", preview: "The agent couldn't understand my request about...", tokens: 1890, turns: 11, sentiment: "negative", flagged: true, time: "15m ago" },
];

const SENTIMENT_COLORS: Record<string, string> = {
  positive: "#22C55E",
  negative: "#EF4444",
  neutral: "var(--muted-foreground)",
};

export function ConversationsCard() {
  return (
    <Card>
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
        <div className="flex items-center gap-2">
          <span style={{ color: "#22C55E" }}>\u22A1</span>
          <span className="text-sm font-semibold text-foreground font-display">Recent Conversations</span>
          <Badge variant="danger">2 flagged</Badge>
        </div>
        <div className="flex gap-1.5">
          {["All", "Flagged", "Negative"].map((f) => (
            <button
              key={f}
              type="button"
              className={cn(
                "px-2 py-[3px] rounded border border-border cursor-pointer font-display transition-colors",
                f === "All"
                  ? "bg-accent text-foreground/80"
                  : "bg-transparent text-faint hover:text-subtle",
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>
      {CONVERSATIONS_RECENT.map((c) => (
        <div
          key={c.id}
          className="px-4 py-2.5 border-b border-input flex items-center gap-3 cursor-pointer transition-colors duration-100 hover:bg-input"
        >
          <div
            className="w-1.5 h-1.5 rounded-full shrink-0"
            style={{ background: SENTIMENT_COLORS[c.sentiment] }}
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-0.5">
              <span className="text-foreground/80 font-medium">{c.agent}</span>
              <span className="text-[9px] text-faint">\u2192</span>
              <span className="text-muted-foreground">{c.user}</span>
              {c.flagged && <Badge variant="danger">flagged</Badge>}
            </div>
            <div className="text-subtle truncate">{c.preview}</div>
          </div>
          <div className="text-right shrink-0">
            <div className="text-muted-foreground">{c.tokens} tok \u00b7 {c.turns} turns</div>
            <div className="text-[9px] text-faint">{c.time}</div>
          </div>
        </div>
      ))}
    </Card>
  );
}
