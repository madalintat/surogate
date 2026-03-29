// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useMemo } from "react";
import { useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { Card } from "@/components/ui/card";
import { cn } from "@/utils/cn";
import {
  REPO_ITEMS,
  TYPE_META,
  VIS_COLORS,
} from "./hub-data";
import type { RepoItem } from "./hub-data";
import { Download, Heart, LayoutGrid } from "lucide-react";

export function HubPage() {
  const navigate = useNavigate();
  const [filterType, setFilterType] = useState<string>("all");
  const [filterSearch, setFilterSearch] = useState("");
  const [filterSort, setFilterSort] = useState("updated");
  const [selectedTag, setSelectedTag] = useState<string | null>(null);

  const filtered = useMemo(() => {
    let items = REPO_ITEMS as RepoItem[];
    if (filterType !== "all") items = items.filter((r) => r.type === filterType);
    if (selectedTag) items = items.filter((r) => r.tags.includes(selectedTag));
    if (filterSearch) {
      const q = filterSearch.toLowerCase();
      items = items.filter(
        (r) =>
          r.name.toLowerCase().includes(q) ||
          r.displayName.toLowerCase().includes(q) ||
          r.description.toLowerCase().includes(q) ||
          r.tags.some((t) => t.includes(q)),
      );
    }
    if (filterSort === "downloads") items = [...items].sort((a, b) => b.downloads - a.downloads);
    else if (filterSort === "likes") items = [...items].sort((a, b) => b.likes - a.likes);
    return items;
  }, [filterType, filterSearch, filterSort, selectedTag]);

  const counts: Record<string, number> = {
    all: REPO_ITEMS.length,
    model: REPO_ITEMS.filter((r) => r.type === "model").length,
    dataset: REPO_ITEMS.filter((r) => r.type === "dataset").length,
    agent: REPO_ITEMS.filter((r) => r.type === "agent").length,
    skill: REPO_ITEMS.filter((r) => r.type === "skill").length,
  };
  const allTags = [...new Set(REPO_ITEMS.flatMap((r) => r.tags))].sort();

  const typeFilters = [
    { id: "all", icon: LayoutGrid, label: "All", count: counts.all, color: "#22C55E" },
    ...Object.entries(TYPE_META).map(([id, m]) => ({
      id,
      icon: m.icon,
      label: m.plural,
      count: counts[id],
      color: m.color,
    })),
  ];

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Hub"
        subtitle={`${counts.model} models · ${counts.dataset} datasets · ${counts.agent} agents · ${counts.skill} skills`}
      />

      {/* filter bar */}
      <div className="px-7 py-2.5 border-b border-line bg-card shrink-0">
        <div className="flex items-center gap-3 mb-2">
          <div className="flex items-center gap-2 px-2.5 py-1.5 bg-input border border-border rounded-md flex-1 max-w-[480px]">
            <span className="text-faint text-[13px]">⌕</span>
            <input
              value={filterSearch}
              onChange={(e) => setFilterSearch(e.target.value)}
              placeholder="Search models, datasets, agents, skills..."
              className="flex-1 bg-transparent border-none outline-none text-foreground font-mono"
            />
            {filterSearch && (
              <button
                type="button"
                onClick={() => setFilterSearch("")}
                className="bg-transparent border-none text-muted-foreground cursor-pointer"
              >
                ✕
              </button>
            )}
          </div>
          <span className="text-faint">{filtered.length} results</span>
          <select
            value={filterSort}
            onChange={(e) => setFilterSort(e.target.value)}
            className="px-2 py-1 rounded border border-border bg-input text-muted-foreground font-display cursor-pointer outline-none"
          >
            <option value="updated">Recently updated</option>
            <option value="downloads">Most downloads</option>
            <option value="likes">Most liked</option>
          </select>
        </div>
        <div className="flex items-center gap-1.5 flex-wrap">
          {typeFilters.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => {
                setFilterType(c.id);
                setSelectedTag(null);
              }}
              className={cn(
                "flex items-center gap-1.5 border px-2.5 py-1 rounded-[5px] cursor-pointer font-display transition-all duration-150",
                filterType === c.id ? "font-semibold" : "font-normal",
              )}
              style={{
                borderColor: filterType === c.id ? `${c.color}33` : "var(--border)",
                background: filterType === c.id ? `${c.color}10` : "transparent",
                color: filterType === c.id ? c.color : "var(--subtle)",
              }}
            >
              <c.icon size={12} />
              {c.label}
              <span className="text-[9px] opacity-60">{c.count}</span>
            </button>
          ))}
          <div className="w-px h-[18px] bg-line mx-0.5" />
          {(selectedTag ? [selectedTag] : allTags.slice(0, 10)).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setSelectedTag(selectedTag === t ? null : t)}
              className="px-[7px] py-[3px] rounded cursor-pointer border font-display"
              style={{
                borderColor: selectedTag === t ? "#22C55E33" : "var(--border)",
                background: selectedTag === t ? "#22C55E10" : "transparent",
                color: selectedTag === t ? "#22C55E" : "var(--muted-foreground)",
              }}
            >
              {t}
            </button>
          ))}
          {selectedTag && (
            <button
              type="button"
              onClick={() => setSelectedTag(null)}
              className="text-success cursor-pointer font-display font-semibold"
            >
              CLEAR
            </button>
          )}
          {!selectedTag && allTags.length > 10 && (
            <span className="text-faint">+{allTags.length - 10}</span>
          )}
        </div>
      </div>

      {/* item list */}
      <div className="flex-1 overflow-y-auto px-7 py-4 pb-10">
        <div className="flex flex-col gap-2">
          {filtered.map((r) => {
            const rtm = TYPE_META[r.type];
            return (
              <div
                key={r.id}
                onClick={() => navigate({ to: `/studio/hub/${r.id}` })}
                className="cursor-pointer"
              >
                <Card className="px-4 py-3.5 transition-all duration-150 hover:border-border">
                  <div className="flex items-start gap-3">
                    <div
                      className="w-9 h-9 rounded-lg shrink-0 flex items-center justify-center text-base border"
                      style={{
                        background: `${rtm.color}10`,
                        borderColor: `${rtm.color}22`,
                        color: rtm.color,
                      }}
                    >
                      <rtm.icon size={18} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5 flex-wrap">
                        <span className="font-medium font-display" style={{ color: r.projectColor }}>{r.project}</span>
                        <span className="text-faint">/</span>
                        <span className="font-bold text-foreground font-display">{r.name}</span>
                        <code className="text-muted-foreground bg-line px-[5px] py-px rounded">{r.version}</code>
                        <span
                          className="text-sm px-[5px] py-px rounded font-semibold uppercase border"
                          style={{ background: `${rtm.color}12`, color: rtm.color, borderColor: `${rtm.color}25` }}
                        >
                          {rtm.label}
                        </span>
                        <span
                          className="text-xs px-[5px] py-px rounded"
                          style={{ background: VIS_COLORS[r.visibility].bg, color: VIS_COLORS[r.visibility].fg }}
                        >
                          {r.visibility}
                        </span>
                        {r.serving && (
                          <span className="text-sm px-1 py-px rounded font-semibold border" style={{ background: "#22C55E12", color: "#22C55E", borderColor: "#22C55E20" }}>
                            SERVING
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-subtle leading-normal mb-1.5 truncate">{r.description}</p>
                      <div className="flex items-center gap-2.5 text-sm text-faint">
                        <span>{r.author}</span>
                        <span>·</span>
                        <span>updated {r.updatedAt}</span>
                        <span>·</span>
                        <span>{r.size}</span>
                        <span>·</span>
                        <span className="flex items-center gap-1"><Download size={12} />{r.downloads}</span>
                        <span>·</span>
                        <span className="flex items-center gap-1"><Heart size={12} />{r.likes}</span>
                        <span className="flex-1" />
                        <div className="flex gap-[3px]">
                          {r.tags.slice(0, 4).map((t) => (
                            <span key={t} className="text-sm px-1 py-px rounded bg-accent text-muted-foreground">{t}</span>
                          ))}
                          {r.tags.length > 4 && <span className="text-sm text-faint">+{r.tags.length - 4}</span>}
                        </div>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            );
          })}
          {filtered.length === 0 && (
            <div className="py-10 text-center text-faint">No repositories match your filters</div>
          )}
        </div>
      </div>
    </div>
  );
}
