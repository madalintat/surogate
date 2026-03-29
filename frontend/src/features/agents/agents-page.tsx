// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { PageHeader } from "@/components/page-header";

export function AgentsPage() {
  return (
    <div className="flex-1 overflow-auto bg-background">
      <PageHeader title="Agents" subtitle="Manage deployed agents, replicas, and configurations." />
      <div className="p-7" />
    </div>
  );
}
