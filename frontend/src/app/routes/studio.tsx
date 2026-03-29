// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Outlet, createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { useInitialData } from "@/hooks/use-initial-data";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const StudioPage = lazy(() =>
  import("@/features/studio/studio-page").then((m) => ({
    default: m.StudioPage,
  })),
);

const AgentsPage = lazy(() =>
  import("@/features/agents/agents-page").then((m) => ({
    default: m.AgentsPage,
  })),
);

const ModelsPage = lazy(() =>
  import("@/features/models/models-page").then((m) => ({
    default: m.ModelsPage,
  })),
);

const SkillsPage = lazy(() =>
  import("@/features/skills/skills-page").then((m) => ({
    default: m.SkillsPage,
  })),
);

const ComputePage = lazy(() =>
  import("@/features/compute/compute-page").then((m) => ({
    default: m.ComputePage,
  })),
);

const MonitoringPage = lazy(() =>
  import("@/features/monitoring/monitoring-page").then((m) => ({
    default: m.MonitoringPage,
  })),
);

const ConversationsPage = lazy(() =>
  import("@/features/conversations/conversations-page").then((m) => ({
    default: m.ConversationsPage,
  })),
);

const EvaluationsPage = lazy(() =>
  import("@/features/evaluations/evaluations-page").then((m) => ({
    default: m.EvaluationsPage,
  })),
);

const PlaygroundPage = lazy(() =>
  import("@/features/playground/playground-page").then((m) => ({
    default: m.PlaygroundPage,
  })),
);

const DatasetsPage = lazy(() =>
  import("@/features/datasets/datasets-page").then((m) => ({
    default: m.DatasetsPage,
  })),
);

const TrainingPage = lazy(() =>
  import("@/features/training/training-page").then((m) => ({
    default: m.TrainingPage,
  })),
);

const HubPage = lazy(() =>
  import("@/features/hub/hub-page").then((m) => ({
    default: m.HubPage,
  })),
);

function StudioLayout() {
  useInitialData();
  return <Outlet />;
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/studio",
  beforeLoad: () => requireAuth(),
  component: StudioLayout,
});

export const studioIndexRoute = createRoute({
  getParentRoute: () => Route,
  path: "/",
  component: StudioPage,
});

export const agentsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/agents",
  component: AgentsPage,
});

export const modelsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/models",
  component: ModelsPage,
});

export const skillsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/skills",
  component: SkillsPage,
});

export const computeRoute = createRoute({
  getParentRoute: () => Route,
  path: "/compute",
  component: ComputePage,
});

export const monitoringRoute = createRoute({
  getParentRoute: () => Route,
  path: "/monitoring",
  component: MonitoringPage,
});

export const conversationsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/conversations",
  component: ConversationsPage,
});

export const evaluationsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/evaluations",
  component: EvaluationsPage,
});

export const playgroundRoute = createRoute({
  getParentRoute: () => Route,
  path: "/playground",
  component: PlaygroundPage,
});

export const datasetsRoute = createRoute({
  getParentRoute: () => Route,
  path: "/datasets",
  component: DatasetsPage,
});

export const trainingRoute = createRoute({
  getParentRoute: () => Route,
  path: "/training",
  component: TrainingPage,
});

export const hubRoute = createRoute({
  getParentRoute: () => Route,
  path: "/hub",
  component: HubPage,
});
