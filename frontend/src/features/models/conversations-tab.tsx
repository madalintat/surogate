// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState, useCallback } from "react";
import { Input } from "@/components/ui/input";
import { ConversationListItem } from "@/components/conversations/conversation-list-item";
import { ConversationDetail, ConversationEmptyState } from "@/components/conversations/conversation-detail";
import type { Conversation, ConversationDetail as ConversationDetailType } from "@/types/conversation";
import * as conversationsApi from "@/api/conversations";
import type { Model } from "./models-data";

export function ConversationsTab({ model }: { model: Model }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedConversation, setSelectedConversation] = useState<ConversationDetailType | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const fetchList = useCallback(async () => {
    try {
      setLoading(true);
      const res = await conversationsApi.listConversations({ deployed_model_id: model.id });
      setConversations(res.conversations);
      setTotal(res.total);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, [model.id]);

  useEffect(() => {
    fetchList();
  }, [fetchList]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedConversation(null);
      return;
    }
    let cancelled = false;
    conversationsApi.getConversation(selectedId).then((detail) => {
      if (!cancelled) setSelectedConversation(detail);
    });
    return () => { cancelled = true; };
  }, [selectedId]);

  const handleDelete = async (conversationId: string) => {
    await conversationsApi.deleteConversation(conversationId);
    setSelectedId(null);
    setSelectedConversation(null);
    fetchList();
  };

  const filtered = conversations.filter((c) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      c.model.toLowerCase().includes(q) ||
      c.projectName.toLowerCase().includes(q) ||
      c.runName.toLowerCase().includes(q)
    );
  });

  return (
    <div className="flex overflow-hidden rounded-lg border border-border" style={{ height: "calc(100vh - 280px)" }}>
      {/* Conversation list */}
      <div className="w-[340px] min-w-[340px] border-r border-border flex flex-col">
        <div className="px-3 py-2.5 border-b border-border">
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search conversations..."
            className="h-7 text-xs"
          />
        </div>
        <div className="flex-1 overflow-y-auto">
          {filtered.map((c) => (
            <ConversationListItem
              key={c.id}
              convo={c}
              selected={selectedId === c.id}
              onSelect={() => setSelectedId(c.id)}
            />
          ))}
          {filtered.length === 0 && !loading && (
            <div className="py-8 text-center text-muted-foreground/30 text-xs">
              {conversations.length === 0
                ? "No conversations for this model"
                : "No conversations match search"}
            </div>
          )}
          {filtered.length === 0 && loading && (
            <div className="py-8 text-center text-muted-foreground/30 text-xs">
              Loading...
            </div>
          )}
        </div>
        {total > 0 && (
          <div className="px-3 py-1.5 border-t border-border text-[9px] text-muted-foreground/30">
            {total} conversation{total !== 1 ? "s" : ""}
          </div>
        )}
      </div>

      {/* Detail */}
      {selectedConversation ? (
        <ConversationDetail convo={selectedConversation} onDelete={handleDelete} />
      ) : (
        <ConversationEmptyState />
      )}
    </div>
  );
}
