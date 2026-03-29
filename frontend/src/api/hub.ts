// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";

import type { RepositoryList, Repository, RefList, Ref, CommitList, Commit, ObjectStatsList, ObjectStats } from "@/types/hub";

// ============ Repositories ============

export async function listRepositories(prefix?: string): Promise<RepositoryList> {
    const params = new URLSearchParams();
    if (prefix) {
        params.append("prefix", prefix);
    }
    const response = await authFetch(`/api/hub/repositories?${params}`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? "Failed to fetch repositories");
      }
    
      return (await response.json()) as RepositoryList;
}

export async function getRepository(repository: string): Promise<Repository> {
    const response = await authFetch(`/api/hub/repositories/${repository}`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch repository '${repository}'`);
      }
    
      return (await response.json()) as Repository;
}

export async function deleteRepository(repository: string): Promise<{ success: boolean }> {
    const response = await authFetch(`/api/hub/repositories/${repository}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to delete repository '${repository}'`);
      }

      return (await response.json()) as { success: boolean };
}

export async function createRepository(repoName: string, repoType: string): Promise<Repository> {
    const params = new URLSearchParams({ repo_name: repoName, repo_type: repoType });
    const response = await authFetch(`/api/hub/repositories?${params}`, {
        method: "POST",
    });
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to create repository '${repoName}'`);
      }
    
      return (await response.json()) as Repository;
}

// ============ Branches ============

export async function listBranches(repository: string): Promise<RefList> {
    const response = await authFetch(`/api/hub/repositories/${repository}/branches`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch branches for '${repository}'`);
    }
    return (await response.json()) as RefList;
}

export async function createBranch(repository: string, branch: string, source?: string): Promise<string> {
    const params = new URLSearchParams({ branch });
    if (source) params.append("source", source);
    const response = await authFetch(`/api/hub/repositories/${repository}/branches?${params}`, {
        method: "POST",
    });
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to create branch '${branch}'`);
    }
    return (await response.json()) as string;
}

export async function getBranch(repository: string, branch: string): Promise<Ref> {
    const response = await authFetch(`/api/hub/repositories/${repository}/branches/${branch}`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch branch '${branch}'`);
    }
    return (await response.json()) as Ref;
}

export async function deleteBranch(repository: string, branch: string): Promise<{ success: boolean }> {
    const response = await authFetch(`/api/hub/repositories/${repository}/branches/${branch}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to delete branch '${branch}'`);
    }
    return (await response.json()) as { success: boolean };
}

// ============ Tags ============

export async function listTags(repository: string): Promise<RefList> {
    const response = await authFetch(`/api/hub/repositories/${repository}/tags`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch tags for '${repository}'`);
    }
    return (await response.json()) as RefList;
}

export async function createTag(repository: string, tag: string, commit?: string): Promise<string> {
    const params = new URLSearchParams({ tag });
    if (commit) params.append("commit", commit);
    const response = await authFetch(`/api/hub/repositories/${repository}/tags?${params}`, {
        method: "POST",
    });
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to create tag '${tag}'`);
    }
    return (await response.json()) as string;
}

export async function getTag(repository: string, tag: string): Promise<Ref> {
    const response = await authFetch(`/api/hub/repositories/${repository}/tags/${tag}`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch tag '${tag}'`);
    }
    return (await response.json()) as Ref;
}

export async function deleteTag(repository: string, tag: string): Promise<{ success: boolean }> {
    const response = await authFetch(`/api/hub/repositories/${repository}/tags/${tag}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to delete tag '${tag}'`);
    }
    return (await response.json()) as { success: boolean };
}

// ============ Commits ============

export async function listCommits(repository: string, ref: string): Promise<CommitList> {
    const response = await authFetch(`/api/hub/repositories/${repository}/refs/${ref}/commits`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch commits for '${ref}'`);
    }
    return (await response.json()) as CommitList;
}

export async function getCommit(repository: string, commitId: string): Promise<Commit> {
    const response = await authFetch(`/api/hub/repositories/${repository}/commits/${commitId}`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch commit '${commitId}'`);
    }
    return (await response.json()) as Commit;
}

// ============ Objects ============

export async function listObjects(repository: string, ref: string, prefix?: string): Promise<ObjectStatsList> {
    const params = new URLSearchParams();
    if (prefix) params.append("prefix", prefix);
    const response = await authFetch(`/api/hub/repositories/${repository}/refs/${ref}/objects/ls?${params}`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to list objects for '${ref}'`);
    }
    return (await response.json()) as ObjectStatsList;
}

export async function getObject(repository: string, ref: string, path: string): Promise<ObjectStats> {
    const params = new URLSearchParams({ path });
    const response = await authFetch(`/api/hub/repositories/${repository}/refs/${ref}/objects?${params}`);
    if (!response.ok) {
        const err = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(err?.detail ?? `Failed to fetch object '${path}'`);
    }
    return (await response.json()) as ObjectStats;
}
