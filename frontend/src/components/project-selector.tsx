// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CreateProjectDialog } from "@/components/create-project-dialog";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";
import { Plus } from "lucide-react";
import { useState } from "react";

export function ProjectSelector() {
  const projects = useAppStore((s) => s.projects);
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const setActiveProject = useAppStore((s) => s.setActiveProject);
  const [createOpen, setCreateOpen] = useState(false);

  return (
    <div className="flex gap-1">
      {projects.map((p) => (
        <button
          key={p.id}
          type="button"
          onClick={() => setActiveProject(p.id)}
          className={cn(
            "px-2.5 py-1 rounded font-display font-medium cursor-pointer border transition-all duration-150",
            activeProjectId === p.id
              ? ""
              : "border-border bg-transparent text-muted-foreground hover:text-foreground",
          )}
          style={
            activeProjectId === p.id
              ? {
                  color: p.color,
                  borderColor: `${p.color}44`,
                  backgroundColor: `${p.color}15`,
                }
              : undefined
          }
        >
          {p.name}
        </button>
      ))}
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => setCreateOpen(true)}
            className="px-2 py-1 rounded border border-dashed border-border bg-transparent text-faint cursor-pointer hover:text-muted-foreground"
          >
            <Plus size={14} />
          </button>
        </TooltipTrigger>
        <TooltipContent>Create a new Project</TooltipContent>
      </Tooltip>
      <CreateProjectDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}
