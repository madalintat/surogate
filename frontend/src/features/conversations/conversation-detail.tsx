// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useAppStore } from "@/stores/app-store";
import { ThreadTab } from "./thread-tab";
import { TrajectoryTab } from "./trajectory-tab";
import { MetadataTab } from "./metadata-tab";
import type { ConversationDetail as ConversationDetailType } from "./conversations-data";

export function ConversationDetail({
  convo,
}: {
  convo: ConversationDetailType;
}) {
  const deleteConversation = useAppStore((s) => s.deleteConversation);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-sm font-bold text-foreground font-display tracking-tight">
                {convo.model || convo.runName}
              </span>
              <span className="text-[11px] text-muted-foreground/30">&harr;</span>
              <span className="text-[12px] text-muted-foreground font-medium">
                {convo.projectName}
              </span>
              <code className="text-[9px] text-muted-foreground/30 bg-muted px-1.5 py-px rounded">
                {convo.id.substring(0, 8)}...
              </code>
              {convo.hasCompaction && (
                <span
                  className="text-[8px] px-1.5 py-px rounded font-semibold font-display border"
                  style={{
                    background: "#F59E0B12",
                    color: "#F59E0B",
                    borderColor: "#F59E0B30",
                  }}
                >
                  COMPACTED
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 text-[10px] text-muted-foreground/30">
              <span>{convo.turnCount} turns</span>
              <span>{convo.duration}</span>
              {convo.avgLatencyMs != null && (
                <span>avg {Math.round(convo.avgLatencyMs)}ms</span>
              )}
              <span>
                {convo.tokensIn + convo.tokensOut} tokens
              </span>
            </div>
          </div>

          <div className="shrink-0">
            <Button
              variant="destructive"
              size="xs"
              onClick={() => setShowDeleteConfirm(true)}
            >
              Delete
            </Button>
          </div>
        </div>
      </div>

      <ConfirmDialog
        open={showDeleteConfirm}
        title="Delete conversation"
        description={<>This will permanently delete all <strong>{convo.turnCount} turns</strong> in this conversation. This action cannot be undone.</>}
        confirmLabel="Delete"
        variant="destructive"
        onConfirm={async () => {
          await deleteConversation(convo.id);
          setShowDeleteConfirm(false);
        }}
        onCancel={() => setShowDeleteConfirm(false)}
      />

      {/* Tabs */}
      <Tabs
        defaultValue="thread"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="px-5 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="thread">Thread</TabsTrigger>
            <TabsTrigger value="trajectory">Trajectory</TabsTrigger>
            <TabsTrigger value="metadata">Metadata</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          <TabsContent value="thread" className="mt-0">
            <ThreadTab convo={convo} />
          </TabsContent>
          <TabsContent value="trajectory" className="mt-0">
            <TrajectoryTab convo={convo} />
          </TabsContent>
          <TabsContent value="metadata" className="mt-0">
            <MetadataTab convo={convo} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

export function ConversationEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x2B21;</div>
        <div className="font-display text-sm">
          Select a conversation to inspect
        </div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          Conversations are automatically captured from chat completion requests
          flowing through the proxy.
        </div>
      </div>
    </div>
  );
}
