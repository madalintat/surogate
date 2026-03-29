// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { PageHeader } from "@/components/page-header";

export function SkillsPage() {
  return (
    <div className="flex-1 overflow-auto bg-background">
      <PageHeader title="Skills & Tools" subtitle="Reusable skills, MCP tools, and function definitions." />
      <div className="p-7" />
    </div>
  );
}
