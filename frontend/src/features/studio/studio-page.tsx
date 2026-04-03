// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { PageHeader } from "@/components/page-header";
import { MetricCards } from "./metric-cards";
import { AgentsCard } from "./agents-card";
import { ServingModelsCard } from "./serving-models-card";
import { TrainingJobsCard } from "./training-jobs-card";
import { ConversationsCard } from "./conversations-card";
import { QuickActionsCard } from "./quick-actions-card";
import { ClusterHealthCard } from "./cluster-health-card";
import { ActivityCard } from "./activity-card";

export function StudioPage() {
  const now = new Date();

  return (
    <div className="flex-1 overflow-auto bg-background">
      <PageHeader
        title="Dashboard"
        subtitle={
          <>
            {now.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
            {" \u00b7 "}
            <span className="text-primary">All systems operational</span>
          </>
        }
      />

      <div className="p-7 pb-10">
        <MetricCards />

        <div className="grid grid-cols-[1fr_320px] gap-4">
          {/* left column */}
          <div className="flex flex-col gap-4">
            <AgentsCard />
            <div className="grid grid-cols-2 gap-4">
              <ServingModelsCard />
              <TrainingJobsCard />
            </div>
            <ConversationsCard />
          </div>

          {/* right column */}
          <div className="flex flex-col gap-4">
            <QuickActionsCard />
            <ClusterHealthCard />
            <ActivityCard />
          </div>
        </div>
      </div>
    </div>
  );
}
