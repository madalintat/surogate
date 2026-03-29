// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useRef } from "react";
import { fetchProjects } from "@/api/project";
import { useAppStore } from "@/stores/app-store";

export function useInitialData() {
  const loaded = useRef(false);
  const setProjects = useAppStore((s) => s.setProjects);
  const setActiveProject = useAppStore((s) => s.setActiveProject);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;

    async function load() {
      try {
        const projects = await fetchProjects();
        setProjects(projects);
        if (projects.length > 0) {
          setActiveProject(projects[0].id);
        }
      } catch (err) {
        console.error("Failed to load initial data:", err);
      }
    }

    void load();
  }, [setProjects, setActiveProject]);
}
