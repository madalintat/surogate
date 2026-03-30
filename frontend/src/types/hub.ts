export const RepositoryType = {
  MODEL: "model",
  DATASET: "dataset",
  AGENT: "agent",
  SKILL: "skill",
} as const;

export type RepositoryType = typeof RepositoryType[keyof typeof RepositoryType];

export interface Pagination {
  has_more: boolean;
  next_offset: string;
  results: number;
  max_per_page: number;
}

export interface Repository {
  id: string;
  creation_date: number;
  default_branch: string;
  storage_id?: string;
  storage_namespace: string;
  read_only?: boolean;
  metadata?: Record<string, string>;
}

export interface RepositoryList {
  pagination: Pagination;
  results: Repository[];
}

export interface RepositoryCreation {
  name: string;
  storage_namespace: string;
  metadata?: Record<string, string>;
}

export interface BranchCreation {
  name: string;
  source: string;
  force?: boolean;
  hidden?: boolean;
}

export interface Ref {
  id: string;
  commit_id: string;
}

export interface RefList {
  pagination: Pagination;
  results: Ref[];
}

export interface Commit {
  id: string;
  parents: string[];
  committer: string;
  message: string;
  creation_date: number;
  meta_range_id: string;
  metadata?: Record<string, string>;
  generation?: number;
  version?: number;
}

export interface CommitList {
  pagination: Pagination;
  results: Commit[];
}

export interface ObjectStats {
  path: string;
  path_type: "common_prefix" | "object";
  physical_address: string;
  physical_address_expiry?: number;
  checksum: string;
  size_bytes?: number;
  mtime: number;
  metadata?: Record<string, string>;
  content_type?: string;
}

export interface ObjectStatsList {
  pagination: Pagination;
  results: ObjectStats[];
}

