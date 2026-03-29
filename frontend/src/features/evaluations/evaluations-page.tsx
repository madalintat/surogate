// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { PageHeader } from "@/components/page-header";

export function EvaluationsPage() {
  return (
    <div className="flex-1 overflow-auto bg-background">
      <PageHeader title="Evaluations" subtitle="Run benchmarks, custom evals, and track model quality." />
      <div className="p-7" />
    </div>
  );
}
