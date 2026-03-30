// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Settings, Sun, Moon, LogOut } from "lucide-react";
import { useTheme } from "@/hooks/use-theme";
import { logout } from "@/api/auth";
import { ProjectSelector } from "@/components/project-selector";
import { TYPE_META } from "./hub-data";
import type { RepoType } from "./hub-data";
import type { Repository } from "@/types/hub";

interface RepoHeaderProps {
  repo: Repository;
}

export function RepoHeader({ repo }: RepoHeaderProps) {
  const { isDark, toggle } = useTheme();

  const repoType = repo.metadata?.type as RepoType | undefined;
  const meta = repoType ? TYPE_META[repoType] : null;
  const color = meta?.color ?? "#6B7280";

  return (
    <header className="px-7 py-3 border-b border-line flex items-center justify-between bg-card sticky top-0 z-5">
      <div className="flex items-center gap-3 min-w-0">
        {meta && (
          <div
            className="w-9 h-9 rounded-lg shrink-0 flex items-center justify-center border"
            style={{ background: `${color}10`, borderColor: `${color}22`, color }}
          >
            <meta.icon size={18} />
          </div>
        )}
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="font-display text-[17px] font-bold text-foreground tracking-tight truncate">
              {repo.id}
            </h1>
            {meta && (
              <span
                className="text-xs px-1.5 py-px rounded font-semibold uppercase border shrink-0"
                style={{ background: `${color}12`, color, borderColor: `${color}25` }}
              >
                {meta.label}
              </span>
            )}
          </div>
          {repo.metadata?.description && (
            <p className="text-muted-foreground text-sm mt-px truncate">{repo.metadata.description}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        <ProjectSelector />
        <div className="w-px h-5 bg-line" />
        <button
          type="button"
          className="bg-transparent border-none text-muted-foreground cursor-pointer hover:text-foreground"
        >
          <Settings size={16} />
        </button>
        <button
          type="button"
          onClick={toggle}
          className="bg-transparent border-none text-muted-foreground cursor-pointer hover:text-foreground"
        >
          {isDark ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        <button
          type="button"
          onClick={() => { logout(); window.location.href = "/login"; }}
          className="bg-transparent border-none text-muted-foreground cursor-pointer hover:text-foreground"
        >
          <LogOut size={16} />
        </button>
      </div>
    </header>
  );
}
