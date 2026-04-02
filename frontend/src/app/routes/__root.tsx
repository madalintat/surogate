// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Navbar } from "@/components/navbar";
import { useMonitorSocket } from "@/hooks/use-monitor-socket";
import {
  Outlet,
  createRootRoute,
  useRouterState,
} from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { Suspense } from "react";
import { AppProvider } from "../provider";

const HIDDEN_NAVBAR_ROUTES = ["/onboarding", "/login", "/change-password"];

function RootLayout() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const hideNavbar = HIDDEN_NAVBAR_ROUTES.includes(pathname);
  useMonitorSocket();

  return (
    <AppProvider>
      <div className="flex h-screen overflow-hidden font-mono bg-background text-foreground">
        {!hideNavbar && <Navbar />}
        <AnimatePresence initial={false}>
          <motion.div
            key={pathname}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.15 }}
            className="flex-1 flex flex-col overflow-hidden"
          >
            <Suspense fallback={null}>
              <Outlet />
            </Suspense>
          </motion.div>
        </AnimatePresence>
      </div>
    </AppProvider>
  );
}

export const Route = createRootRoute({
  beforeLoad: ({ location }) => {
    // Extra logic to handle the case where the user tries to access a route
  },
  component: RootLayout,
});
