// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

/* ── enums ── */

export type UserRole = "admin" | "skill_author" | "devops" | "developer";

export type ProjectStatus = "active" | "archived";

export type ProjectMemberRole = "owner" | "editor" | "viewer";

/* ── entities ── */

export interface User {
  id: string;
  username: string;
  name: string;
  email: string | null;
  roles: UserRole[];
  avatar_url: string | null;
  last_login_at: string | null;
  default_project_id: string | null;
  created_at: string;
}

export interface Project {
  id: string;
  name: string;
  namespace: string;
  color: string;
  status: ProjectStatus;
  created_by_id: string;
  created_at: string;
}

export interface ProjectMember {
  project_id: string;
  user_id: string;
  role: ProjectMemberRole;
  added_at: string;
  added_by_id: string;
}
