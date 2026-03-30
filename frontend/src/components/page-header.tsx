// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { ReactNode } from "react";
import { Settings, Sun, Moon, LogOut } from "lucide-react";
import { useTheme } from "@/hooks/use-theme";
import { logout } from "@/api/auth";
import { ProjectSelector } from "@/components/project-selector";

interface PageHeaderProps {
  title: string;
  subtitle?: ReactNode;
}

export function PageHeader({ title, subtitle }: PageHeaderProps) {
  const { isDark, toggle } = useTheme();

  return (
    <header className="px-7 py-3 border-b border-line flex items-center justify-between bg-card sticky top-0 z-5">
      <div>
        <h1 className="font-display text-[17px] font-bold text-foreground tracking-tight">
          {title}
        </h1>
        {subtitle && (
          <p className="text-muted-foreground mt-px">{subtitle}</p>
        )}
      </div>
      <div className="flex items-center gap-3">
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
