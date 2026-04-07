// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { ConversationDetail } from "./conversations-data";

function DetailRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-1 border-b border-border/50">
      <span className="text-[10px] text-muted-foreground/40">{label}</span>
      <span className="text-[10px] text-foreground/70">{value}</span>
    </div>
  );
}

export function MetadataTab({ convo }: { convo: ConversationDetail }) {
  return (
    <div className="animate-in fade-in duration-150">
      <div className="grid grid-cols-2 gap-4">
        {/* Conversation Details */}
        <section className="bg-muted/40 border border-border rounded-lg p-3.5">
          <div className="text-xs font-semibold text-foreground font-display mb-3">
            Conversation Details
          </div>
          <DetailRow label="ID" value={convo.id} />
          <DetailRow label="Model" value={convo.model} />
          <DetailRow label="Project" value={convo.projectName} />
          <DetailRow label="Service" value={convo.runName} />
          <DetailRow label="Started" value={convo.startedAt ?? "\u2014"} />
          <DetailRow label="Last Turn" value={convo.lastTurnAt ?? "\u2014"} />
          <DetailRow label="Duration" value={convo.duration || "\u2014"} />
          <DetailRow label="Turns" value={convo.turnCount} />
          <DetailRow label="Compacted" value={convo.hasCompaction ? "Yes" : "No"} />
        </section>

        {/* Token & Performance */}
        <section className="bg-muted/40 border border-border rounded-lg p-3.5">
          <div className="text-xs font-semibold text-foreground font-display mb-3">
            Token & Performance
          </div>
          <DetailRow label="Tokens In" value={convo.tokensIn.toLocaleString()} />
          <DetailRow label="Tokens Out" value={convo.tokensOut.toLocaleString()} />
          <DetailRow label="Total Tokens" value={convo.totalTokens.toLocaleString()} />
          <DetailRow
            label="Avg Latency"
            value={convo.avgLatencyMs != null ? `${Math.round(convo.avgLatencyMs)}ms` : "\u2014"}
          />
        </section>
      </div>

      {/* System Prompt */}
      {convo.systemPrompt && (
        <section className="bg-muted/40 border border-border rounded-lg p-3.5 mt-4">
          <div className="text-xs font-semibold text-foreground font-display mb-2">
            System Prompt
          </div>
          <div className="text-[11px] text-foreground/70 leading-relaxed whitespace-pre-wrap font-mono">
            {convo.systemPrompt}
          </div>
        </section>
      )}

      {/* Turn Hashes */}
      <section className="bg-muted/40 border border-border rounded-lg p-3.5 mt-4">
        <div className="text-xs font-semibold text-foreground font-display mb-2.5">
          Turn Chain Hashes
        </div>
        <div className="space-y-1">
          {convo.turns.map((turn, i) => (
            <div key={turn.id} className="flex items-center gap-2 text-[9px] font-mono">
              <span className="text-muted-foreground/40 w-6 text-right">{i + 1}</span>
              <span className="text-muted-foreground/60" title={`state: ${turn.stateHash}`}>
                {turn.stateHash.substring(0, 16)}...
              </span>
              {turn.parentHash && (
                <>
                  <span className="text-muted-foreground/20">&larr;</span>
                  <span className="text-muted-foreground/40" title={`parent: ${turn.parentHash}`}>
                    {turn.parentHash.substring(0, 16)}...
                  </span>
                </>
              )}
              {turn.compacted && (
                <span
                  className="text-[7px] px-1 py-px rounded font-semibold font-display"
                  style={{ background: "#F59E0B12", color: "#F59E0B" }}
                >
                  HEALED
                </span>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
