// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState } from "react";
import { PageHeader } from "@/components/page-header";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { useAppStore } from "@/stores/app-store";
import { ConversationListItem } from "./conversation-list-item";
import { ConversationDetail, ConversationEmptyState } from "./conversation-detail";

export function ConversationsPage() {
  const conversations = useAppStore((s) => s.conversations);
  const conversationsTotal = useAppStore((s) => s.conversationsTotal);
  const conversationsLoading = useAppStore((s) => s.conversationsLoading);
  const selectedConversation = useAppStore((s) => s.selectedConversation);
  const fetchConversations = useAppStore((s) => s.fetchConversations);
  const fetchConversation = useAppStore((s) => s.fetchConversation);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filterModel, setFilterModel] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch on mount
  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  // Fetch detail when selection changes
  useEffect(() => {
    if (selectedId) {
      fetchConversation(selectedId);
    }
  }, [selectedId, fetchConversation]);

  // Derive unique models for filter
  const models = [...new Set(conversations.map((c) => c.model).filter(Boolean))];

  // Client-side filtering
  const filtered = conversations.filter((c) => {
    if (filterModel !== "all" && c.model !== filterModel) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      if (
        !c.model.toLowerCase().includes(q) &&
        !c.projectName.toLowerCase().includes(q) &&
        !c.runName.toLowerCase().includes(q)
      )
        return false;
    }
    return true;
  });

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Conversations"
        subtitle={
          <>
            {conversationsTotal} conversations
            {conversationsLoading && " (loading...)"}
          </>
        }
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Conversation list (left) */}
        <div className="w-[420px] min-w-[420px] border-r border-border flex flex-col">
          {/* Search + filters */}
          <div className="px-3.5 py-3 border-b border-border space-y-2.5">
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search model, project, service..."
              className="h-8 text-xs"
            />
            <div className="flex gap-1 flex-wrap">
              <Select value={filterModel} onValueChange={setFilterModel}>
                <SelectTrigger
                  size="sm"
                  className="h-6 text-[10px] px-1.5 font-display text-muted-foreground"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Models</SelectItem>
                  {models.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* List */}
          <div className="flex-1 overflow-y-auto">
            {filtered.map((c) => (
              <ConversationListItem
                key={c.id}
                convo={c}
                selected={selectedId === c.id}
                onSelect={() => setSelectedId(c.id)}
              />
            ))}
            {filtered.length === 0 && !conversationsLoading && (
              <div className="py-8 text-center text-muted-foreground/30 text-xs">
                No conversations match filters
              </div>
            )}
            {filtered.length === 0 && conversationsLoading && (
              <div className="py-8 text-center text-muted-foreground/30 text-xs">
                Loading...
              </div>
            )}
          </div>
        </div>

        {/* Detail (right) */}
        {selectedConversation ? (
          <ConversationDetail convo={selectedConversation} />
        ) : (
          <ConversationEmptyState />
        )}
      </div>
    </div>
  );
}
