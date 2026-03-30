// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ArrowLeft, Loader2, Trash2, Upload } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { cn } from "@/utils/cn";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { toast } from "sonner";
import { useHubStore } from "./hub-store";
import { uploadObject } from "@/api/hub";
import { UploadDialog } from "./upload-dialog";
import { RepoHeader } from "./repo-header";
import { FileBrowser } from "./file-browser";
import { ObjectRenderer } from "./object-renderer";
import { RepoInfoTab } from "./repo-info-tab";
import { CommitsTab } from "./commits-tab";
import { BranchesTab } from "./branches-tab";
import { TagsTab } from "./tags-tab";
import { TYPE_META } from "./hub-data";
import type { RepoType } from "./hub-data";

export function RepoDetailPage({ repoId }: { repoId: string }) {
  const navigate = useNavigate();
  const {
    currentRepo: repo,
    branches,
    tags,
    commits,
    objects,
    loading,
    error,
    fetchRepository,
    fetchBranches,
    fetchTags,
    fetchCommits,
    fetchObjects,
    deleteRepository,
  } = useHubStore();

  const [detailTab, setDetailTab] = useState("info");
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState("");
  const [selectedObject, setSelectedObject] = useState<import("@/types/hub").ObjectStats | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showUpload, setShowUpload] = useState(false);

  useEffect(() => {
    void fetchRepository(repoId);
    void fetchBranches(repoId);
    void fetchTags(repoId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repoId]);

  // Fetch commits once we know the default branch
  useEffect(() => {
    if (!repo) return;
    const ref = selectedRef ?? repo.default_branch;
    void fetchCommits(repoId, ref);
  }, [repo, selectedRef, repoId]);

  // Fetch objects when ref or path changes
  useEffect(() => {
    if (!repo) return;
    const ref = selectedRef ?? repo.default_branch;
    void fetchObjects(repoId, ref, currentPath || undefined);
  }, [repo, selectedRef, repoId, currentPath]);

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

  const handleUploadFiles = async (files: File[], uploadPath: string, onProgress: (name: string, percent: number) => void) => {
    if (!repo) return;
    const ref = selectedRef ?? repo.default_branch;
    for (const file of files) {
      const filePath = `${uploadPath}${file.name}`;
      await uploadObject(repoId, ref, filePath, file, (percent) => {
        onProgress(file.name, percent);
      });
    }
    setCurrentPath(uploadPath);
    void fetchObjects(repoId, ref, uploadPath || undefined);
  };

  if (loading && !repo) {
    return (
      <div className="flex-1 flex items-center justify-center bg-background">
        <Loader2 className="animate-spin text-faint" size={24} />
      </div>
    );
  }

  if (error && !repo) {
    return (
      <div className="flex-1 overflow-auto bg-background">
        <PageHeader title="Repository Not Found" />
        <div className="p-7 text-center text-destructive">{error}</div>
      </div>
    );
  }

  if (!repo) return null;

  const repoType = repo.metadata?.type as RepoType | undefined;
  const meta = repoType ? TYPE_META[repoType] : null;
  const activeRef = selectedRef ?? repo.default_branch;
  const repoBranches = branches[repoId] ?? [];
  const repoTags = tags[repoId] ?? [];
  const repoCommits = commits[`${repoId}:${activeRef}`] ?? [];
  const repoObjects = objects[`${repoId}:${activeRef}`] ?? [];
  const repoTagsList = repo.metadata?.tags?.split(",").map((t) => t.trim()).filter(Boolean) ?? [];

  const detailTabs = [
    { id: "info", label: "Info" },
    { id: "files", label: `Files (${repoObjects.length})` },
    { id: "commits", label: `Commits (${repoCommits.length})` },
    { id: "branches", label: `Branches (${repoBranches.length})` },
    { id: "tags", label: `Tags (${repoTags.length})` },
  ];

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <RepoHeader repo={repo} />

      <div className="flex-1 overflow-y-auto">
        {/* action bar */}
        <div className="px-7 py-3 border-b border-line flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate({ to: "/studio/hub" })}
            className="px-3 py-1.5 rounded-md border border-border bg-input text-muted-foreground cursor-pointer font-display flex items-center gap-1.5"
          >
            <ArrowLeft size={14} />
            Back to Hub
          </button>

          {/* ref selector */}
          <select
            value={activeRef}
            onChange={(e) => setSelectedRef(e.target.value)}
            className="px-2 py-1.5 rounded-md border border-border bg-input text-foreground font-display cursor-pointer outline-none"
          >
            <optgroup label="Branches">
              {repoBranches.map((b) => (
                <option key={`b-${b.id}`} value={b.id}>{b.id}</option>
              ))}
            </optgroup>
            <optgroup label="Tags">
              {repoTags.map((t) => (
                <option key={`t-${t.id}`} value={t.id}>{t.id}</option>
              ))}
            </optgroup>
          </select>

          {detailTab === "files" && (
            <button
              type="button"
              onClick={() => setShowUpload(true)}
              className="ml-auto px-3 py-1.5 rounded-md border border-success/30 bg-success/5 text-success cursor-pointer font-display hover:bg-success/10 transition-colors flex items-center gap-1.5"
            >
              <Upload size={14} />
              Upload
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowDeleteConfirm(true)}
            className={cn(
              "px-3 py-1.5 rounded-md border border-destructive/30 bg-destructive/5 text-destructive cursor-pointer font-display hover:bg-destructive/10 transition-colors flex items-center gap-1.5",
              detailTab !== "files" && "ml-auto",
            )}
          >
            <Trash2 size={14} />
            Delete
          </button>
        </div>

        {/* delete confirmation dialog */}
        <ConfirmDialog
          open={showDeleteConfirm}
          title="Delete Repository"
          description={<>Are you sure you want to delete <span className="font-semibold text-foreground">{repoId}</span>? This action cannot be undone.</>}
          confirmLabel="Delete"
          confirmIcon={<Trash2 size={14} className="mr-1.5" />}
          onConfirm={handleDelete}
          onCancel={() => setShowDeleteConfirm(false)}
        />

        {/* upload dialog */}
        {showUpload && (
          <UploadDialog
            destinationPath={currentPath}
            onUpload={handleUploadFiles}
            onClose={() => setShowUpload(false)}
          />
        )}

        {/* tabs */}
        <div className="flex px-7 border-b border-line bg-card sticky top-0 z-1">
          {detailTabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => { setDetailTab(t.id); setSelectedObject(null); }}
              className={cn(
                "px-4 py-2.5 border-none cursor-pointer bg-transparent font-display border-b-2",
                detailTab === t.id
                  ? "text-success font-semibold border-b-success"
                  : "text-muted-foreground font-normal border-b-transparent",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="px-7 py-3 text-destructive text-sm">{error}</div>
        )}

        <div className="px-7 py-5">
          {detailTab === "info" && (
            <RepoInfoTab repo={repo} activeRef={activeRef} typeLabel={meta?.label ?? "Unknown"} tags={repoTagsList} />
          )}
          {detailTab === "files" && !selectedObject && (
            <FileBrowser
              repoId={repoId}
              activeRef={activeRef}
              currentPath={currentPath}
              objects={repoObjects}
              onNavigate={(path) => { setCurrentPath(path); setSelectedObject(null); }}
              onSelectObject={setSelectedObject}
            />
          )}
          {detailTab === "files" && selectedObject && (
            <ObjectRenderer
              repoId={repoId}
              activeRef={activeRef}
              object={selectedObject}
              onBack={() => setSelectedObject(null)}
            />
          )}
          {detailTab === "commits" && (
            <CommitsTab commits={repoCommits} />
          )}
          {detailTab === "branches" && (
            <BranchesTab branches={repoBranches} activeRef={activeRef} onSelectRef={setSelectedRef} />
          )}
          {detailTab === "tags" && (
            <TagsTab tags={repoTags} onSelectRef={setSelectedRef} />
          )}
        </div>
      </div>
    </div>
  );
}
