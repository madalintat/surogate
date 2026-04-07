// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";
import type { Conversation, ConversationDetail } from "@/features/conversations/conversations-data";
import * as conversationsApi from "@/api/conversations";

export type ConversationsSlice = {
  conversations: Conversation[];
  conversationsTotal: number;
  selectedConversation: ConversationDetail | null;
  conversationsLoading: boolean;

  fetchConversations: (params?: {
    project_name?: string;
    model?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }) => Promise<void>;
  fetchConversation: (conversationId: string) => Promise<void>;
  deleteConversation: (conversationId: string) => Promise<boolean>;
  clearSelectedConversation: () => void;
};

export const createConversationsSlice: StateCreator<
  AppState,
  [],
  [],
  ConversationsSlice
> = (set, get) => ({
  conversations: [],
  conversationsTotal: 0,
  selectedConversation: null,
  conversationsLoading: false,

  fetchConversations: async (params) => {
    try {
      set({ conversationsLoading: true });
      const res = await conversationsApi.listConversations(params);
      set({
        conversations: res.conversations,
        conversationsTotal: res.total,
        conversationsLoading: false,
      });
      // Refresh selected conversation if it's still in the list
      const sel = get().selectedConversation;
      if (sel) {
        const still = res.conversations.find((c) => c.id === sel.id);
        if (!still) set({ selectedConversation: null });
      }
    } catch (e) {
      set({ conversationsLoading: false, error: (e as Error).message });
    }
  },

  fetchConversation: async (conversationId) => {
    try {
      const detail = await conversationsApi.getConversation(conversationId);
      set({ selectedConversation: detail });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  deleteConversation: async (conversationId) => {
    try {
      await conversationsApi.deleteConversation(conversationId);
      set({ selectedConversation: null });
      await get().fetchConversations();
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  clearSelectedConversation: () => set({ selectedConversation: null }),
});
