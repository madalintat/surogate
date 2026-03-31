// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ArrowLeft, Trash2 } from "lucide-react";
import { cn } from "@/utils/cn";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { toast } from "sonner";
import { useAppStore } from "@/stores/app-store";
import { RepoHeader } from "./repo-header";
import { RepoExplorer } from "./repo-explorer";

export function RepoDetailPage({ repoId }: { repoId: string }) {
  const navigate = useNavigate();
  const { currentRepo: repo, error, deleteRepository } = useAppStore();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const handleDelete = async () => {
    const ok = await deleteRepository(repoId);
    if (ok) {
      navigate({ to: "/studio/hub" });
      toast.success(`Repository "${repoId}" deleted`);
    } else {
      setShowDeleteConfirm(false);
      toast.error("Failed to delete repository", { description: error ?? undefined });
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      {repo?.id === repoId && <RepoHeader repo={repo} />}

      {/* action bar */}
      <div className="px-7 py-3 border-b border-line flex items-center gap-2 shrink-0">
        <button
          type="button"
          onClick={() => navigate({ to: "/studio/hub" })}
          className="px-3 py-1.5 rounded-md border border-border bg-input text-muted-foreground cursor-pointer font-display flex items-center gap-1.5"
        >
          <ArrowLeft size={14} />
          Back to Hub
        </button>

        <button
          type="button"
          onClick={() => setShowDeleteConfirm(true)}
          className={cn(
            "ml-auto px-3 py-1.5 rounded-md border border-destructive/30 bg-destructive/5 text-destructive cursor-pointer font-display hover:bg-destructive/10 transition-colors flex items-center gap-1.5",
          )}
        >
          <Trash2 size={14} />
          Delete
        </button>
      </div>

      <ConfirmDialog
        open={showDeleteConfirm}
        title="Delete Repository"
        description={<>Are you sure you want to delete <span className="font-semibold text-foreground">{repoId}</span>? This action cannot be undone.</>}
        confirmLabel="Delete"
        confirmIcon={<Trash2 size={14} className="mr-1.5" />}
        onConfirm={handleDelete}
        onCancel={() => setShowDeleteConfirm(false)}
      />

      <RepoExplorer repoId={repoId} className="flex-1" />
    </div>
  );
}
