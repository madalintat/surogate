// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { RouterProvider } from "@tanstack/react-router";
import { router } from "./router";

export function App() {
  return <RouterProvider router={router} />;
}