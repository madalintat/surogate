// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useMemo } from "react";
import { Folder, FileText } from "lucide-react";
import { Card } from "@/components/ui/card";
import type { ObjectStats } from "@/types/hub";

function humanSize(bytes: number | undefined): string {
  if (!bytes) return "—";
  const e = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, e)).toFixed(1) + " " + " KMGTP".charAt(e) + "B";
}

function timeAgo(epoch: number): string {
  const seconds = Math.floor(Date.now() / 1000 - epoch);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes > 1 ? "s" : ""} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours > 1 ? "s" : ""} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days > 1 ? "s" : ""} ago`;
}

interface FileBrowserProps {
  repoId: string;
  activeRef: string;
  currentPath: string;
  objects: ObjectStats[];
  onNavigate: (path: string) => void;
  onSelectObject: (object: ObjectStats) => void;
}

export function FileBrowser({ repoId, activeRef, currentPath, objects, onNavigate, onSelectObject }: FileBrowserProps) {
  const sorted = useMemo(() => {
    return [...objects].sort((a, b) => {
      const aDir = a.path_type === "common_prefix" ? 0 : 1;
      const bDir = b.path_type === "common_prefix" ? 0 : 1;
      if (aDir !== bDir) return aDir - bDir;
      return a.path.localeCompare(b.path);
    });
  }, [objects]);

  return (
    <Card>
      {/* breadcrumb URI navigator */}
      <div className="px-4 py-2.5 border-b border-line flex items-center gap-1 text-sm font-mono">
        <span className="font-bold text-foreground">lakefs://</span>
        <button
          type="button"
          onClick={() => onNavigate("")}
          className="text-success cursor-pointer bg-transparent border-none hover:underline"
        >
          {repoId}
        </button>
        <span className="text-faint">/</span>
        <button
          type="button"
          onClick={() => onNavigate("")}
          className="text-success cursor-pointer bg-transparent border-none hover:underline"
        >
          {activeRef}
        </button>
        <span className="text-faint">/</span>
        {currentPath && currentPath.split("/").filter(Boolean).map((part, i, arr) => {
          const pathTo = arr.slice(0, i + 1).join("/") + "/";
          return (
            <span key={pathTo} className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onNavigate(pathTo)}
                className="text-success cursor-pointer bg-transparent border-none hover:underline"
              >
                {part}
              </button>
              <span className="text-faint">/</span>
            </span>
          );
        })}
      </div>

      {/* file list — folders first, then A-Z */}
      {objects.length === 0 ? (
        <div className="py-8 text-center text-faint">No objects in this ref</div>
      ) : (
        <table className="w-full">
          <tbody>
            {sorted.map((o) => {
              const displayName = currentPath
                ? o.path.slice(currentPath.length)
                : o.path;
              const isDir = o.path_type === "common_prefix";
              return (
                <tr
                  key={o.path}
                  className="border-b border-input hover:bg-input/50 transition-colors"
                >
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <span className={isDir ? "text-[#3B82F6]" : "text-muted-foreground"}>
                        {isDir ? <Folder size={14} /> : <FileText size={14} />}
                      </span>
                      {isDir ? (
                        <button
                          type="button"
                          onClick={() => onNavigate(o.path)}
                          className="text-[#3B82F6] cursor-pointer bg-transparent border-none hover:underline"
                        >
                          {displayName}
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onSelectObject(o)}
                          className="text-foreground cursor-pointer bg-transparent border-none hover:underline"
                        >
                          {displayName}
                        </button>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-faint w-32">
                    {isDir ? "—" : humanSize(o.size_bytes)}
                  </td>
                  <td className="px-4 py-2.5 text-faint text-right w-40">
                    {isDir ? "—" : timeAgo(o.mtime)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Card>
  );
}
