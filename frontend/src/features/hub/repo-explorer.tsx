// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect } from "react";
import { Loader2, Upload } from "lucide-react";
import { cn } from "@/utils/cn";
import { useAppStore } from "@/stores/app-store";
import { uploadObject } from "@/api/hub";
import { UploadDialog } from "./upload-dialog";
import { FileBrowser } from "./file-browser";
import { ObjectRenderer } from "./object-renderer";
import { RepoInfoTab } from "./repo-info-tab";
import { CommitsTab } from "./commits-tab";
import { BranchesTab } from "./branches-tab";
import { TagsTab } from "./tags-tab";
import { TYPE_META } from "./hub-data";
import type { RepoType } from "./hub-data";

interface RepoExplorerProps {
  repoId: string;
  className?: string;
}

export function RepoExplorer({ repoId, className }: RepoExplorerProps) {
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
  } = useAppStore();

  const [detailTab, setDetailTab] = useState("files");
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  const [currentPath, setCurrentPath] = useState("");
  const [selectedObject, setSelectedObject] = useState<import("@/types/hub").ObjectStats | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  useEffect(() => {
    void fetchRepository(repoId);
    void fetchBranches(repoId);
    void fetchTags(repoId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repoId]);

  const defaultBranch = repo?.id === repoId ? repo.default_branch : null;
  const repoTagIds = new Set((tags[repoId] ?? []).map((t) => t.id));
  const isTag = selectedRef != null && repoTagIds.has(selectedRef);

  useEffect(() => {
    if (!defaultBranch || isTag) return;
    const ref = selectedRef ?? defaultBranch;
    void fetchCommits(repoId, ref);
  }, [defaultBranch, selectedRef, repoId, isTag]);

  useEffect(() => {
    if (isTag && detailTab === "commits") setDetailTab("files");
  }, [isTag, detailTab]);

  useEffect(() => {
    if (!defaultBranch) return;
    const ref = selectedRef ?? defaultBranch;
    void fetchObjects(repoId, ref, currentPath || undefined);
  }, [defaultBranch, selectedRef, repoId, currentPath]);

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
      <div className={cn("flex-1 flex items-center justify-center", className)}>
        <Loader2 className="animate-spin text-faint" size={24} />
      </div>
    );
  }

  if (error && !repo) {
    return (
      <div className={cn("flex-1 p-6 text-center text-destructive text-sm", className)}>
        {error}
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
    { id: "files", label: `Files (${repoObjects.length})` },
    ...(!isTag ? [{ id: "commits", label: `Commits (${repoCommits.length})` }] : []),
    { id: "branches", label: `Branches (${repoBranches.length})` },
    { id: "tags", label: `Tags (${repoTags.length})` },
    { id: "info", label: "Info" },
  ];

  return (
    <div className={cn("flex flex-col overflow-hidden", className)}>
      {/* toolbar */}
      <div className="px-6 py-2.5 border-b border-border flex items-center gap-2 shrink-0">
        <select
          value={activeRef}
          onChange={(e) => setSelectedRef(e.target.value)}
          className="px-2 py-1 rounded-md border border-border bg-input text-foreground text-xs font-display cursor-pointer outline-none"
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
            className="ml-auto px-2.5 py-1 rounded-md border border-success/30 bg-success/5 text-success text-xs cursor-pointer font-display hover:bg-success/10 transition-colors flex items-center gap-1.5"
          >
            <Upload size={12} />
            Upload
          </button>
        )}
      </div>

      {/* upload dialog */}
      {showUpload && (
        <UploadDialog
          destinationPath={currentPath}
          onUpload={handleUploadFiles}
          onClose={() => setShowUpload(false)}
        />
      )}

      {/* tabs */}
      <div className="flex px-6 border-b border-border shrink-0">
        {detailTabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => { setDetailTab(t.id); setSelectedObject(null); }}
            className={cn(
              "px-3 py-2 border-none cursor-pointer bg-transparent font-display text-xs border-b-2",
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
        <div className="px-6 py-2 text-destructive text-xs">{error}</div>
      )}

      {/* content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
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
  );
}
