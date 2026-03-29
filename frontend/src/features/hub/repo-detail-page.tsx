// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { ArrowLeft, Download, FileText, Heart } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { Card } from "@/components/ui/card";
import { cn } from "@/utils/cn";
import { REPO_ITEMS, TYPE_META, FILE_ICONS } from "./hub-data";

const FILE_TYPE_COLORS: Record<string, string> = {
  weights: "#3B82F6",
  code: "#8B5CF6",
  data: "#22C55E",
  doc: "#6B7585",
};

export function RepoDetailPage({ repoId }: { repoId: string }) {
  const navigate = useNavigate();
  const [detailTab, setDetailTab] = useState("card");

  const item = REPO_ITEMS.find((r) => r.id === repoId);
  if (!item) {
    return (
      <div className="flex-1 overflow-auto bg-background">
        <PageHeader title="Repository Not Found" />
        <div className="p-7 text-center text-muted-foreground">
          No repository found with ID &quot;{repoId}&quot;.
        </div>
      </div>
    );
  }

  const tm = TYPE_META[item.type];

  const detailTabs = [
    {
      id: "card",
      label:
        item.type === "model"
          ? "Model Card"
          : item.type === "dataset"
            ? "Dataset Card"
            : item.type === "agent"
              ? "Agent Card"
              : "Skill Card",
    },
    { id: "files", label: "Files" },
    { id: "commits", label: "Commits" },
    { id: "versions", label: "Versions" },
  ];

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title={item.displayName}
        subtitle={
          <>
            <span style={{ color: item.projectColor }}>{item.project}</span>
            <span className="text-faint mx-1">/</span>
            {item.name}
            <span className="text-faint mx-1">·</span>
            {item.version}
          </>
        }
      />

      <div className="flex-1 overflow-y-auto">
        {/* detail header */}
        <div className="px-7 py-5 border-b border-line">
          <div className="flex items-start gap-4 mb-4">
            <div
              className="w-12 h-12 rounded-lg shrink-0 flex items-center justify-center text-xl border"
              style={{ background: `${tm.color}10`, borderColor: `${tm.color}22`, color: tm.color }}
            >
              <tm.icon size={22} />
            </div>
            <div className="flex-1">
              <p className="text-subtle leading-relaxed">{item.description}</p>
              <div className="flex items-center gap-2 text-faint flex-wrap">
                <span>{item.author}</span>
                <span>·</span>
                <span>updated {item.updatedAt}</span>
                <span>·</span>
                <span>{item.format}</span>
                <span>·</span>
                <span>{item.size}</span>
                <span>·</span>
                <span className="flex items-center gap-1"><Download size={12} />{item.downloads}</span>
                <span>·</span>
                <span className="flex items-center gap-1"><Heart size={12} />{item.likes}</span>
              </div>
            </div>
          </div>

          <div className="flex gap-2 mb-3">
            <button
              type="button"
              onClick={() => navigate({ to: "/studio/hub" })}
              className="px-3 py-1.5 rounded-md border border-border bg-input text-muted-foreground cursor-pointer font-display flex items-center gap-1.5"
            >
              <ArrowLeft size={14} />
              Back to Hub
            </button>

            <button type="button" className="px-3 py-1.5 rounded-md border border-border bg-input text-foreground/80 cursor-pointer font-display">Upload</button>

            {(item.type === "model" || item.type === "agent") && (
              <button type="button" className="px-3 py-1.5 rounded-md border border-border bg-input text-foreground/80 cursor-pointer font-display">Deploy</button>
            )}

            {(item.type === "model" || item.type === "dataset") && (
              <button type="button" className="px-3 py-1.5 rounded-md border border-border bg-input text-foreground/80 cursor-pointer font-display">Import from HF</button>
            )}

            <button type="button" className="ml-auto px-3 py-1.5 rounded-md border border-destructive/30 bg-destructive/5 text-destructive cursor-pointer font-display hover:bg-destructive/10 transition-colors">Delete</button>
          </div>
        </div>

        {/* tabs */}
        <div className="flex px-7 border-b border-line bg-card sticky top-0 z-1">
          {detailTabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setDetailTab(t.id)}
              className={cn(
                "px-4 py-2.5 border-none cursor-pointer bg-transparent font-display border-b-2",
                detailTab === t.id
                  ? "text-success font-semibold border-b-success"
                  : "text-muted-foreground font-normal border-b-transparent",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="px-7 py-5 max-w-[900px]">
          {/* card tab */}
          {detailTab === "card" && (
            <div className="animate-[fade-in_0.15s_ease]">
              {item.card.metrics && (
                <div className="mb-6">
                  <div className="font-semibold text-foreground font-display mb-3">
                    {item.type === "model" ? "Evaluation Results" : item.type === "dataset" ? "Statistics" : item.type === "agent" ? "Performance" : "Usage"}
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {Object.entries(item.card.metrics).map(([k, v]) => (
                      <Card key={k} className="p-3">
                        <div className="text-[9px] text-muted-foreground uppercase tracking-wider mb-1 font-display">
                          {k.replace(/([A-Z])/g, " $1").replace(/-/g, " ")}
                        </div>
                        <div className="text-lg font-bold text-foreground">{v}</div>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {item.card.training && (
                <div className="mb-6">
                  <div className="font-semibold text-foreground font-display mb-3">Training Details</div>
                  <Card className="p-4">
                    {Object.entries(item.card.training).map(([k, v]) => (
                      <div key={k} className="flex justify-between py-1 border-b border-input">
                        <span className="text-faint">{k}</span>
                        <span className="text-foreground/80 max-w-[300px] truncate text-right">{v}</span>
                      </div>
                    ))}
                  </Card>
                </div>
              )}

              <div className="mb-6">
                <div className="font-semibold text-foreground font-display mb-3">Repository Info</div>
                <Card className="p-4">
                  {[
                    { label: "Type", value: tm.label },
                    { label: "Author", value: item.author },
                    { label: "Project", value: item.project },
                    { label: "Visibility", value: item.visibility },
                    { label: "License", value: item.license },
                    { label: "Format", value: item.format },
                    { label: "Size", value: item.size },
                    { label: "Created", value: item.createdAt },
                    { label: "Updated", value: item.updatedAt },
                    { label: "Downloads", value: String(item.downloads) },
                    { label: "Likes", value: String(item.likes) },
                    ...(item.baseModel ? [{ label: "Base Model", value: item.baseModel.split("/").pop()! }] : []),
                    ...(item.trainingRun ? [{ label: "Training Run", value: item.trainingRun }] : []),
                  ].map((f) => (
                    <div key={f.label} className="flex justify-between py-1 border-b border-input">
                      <span className="text-faint">{f.label}</span>
                      <span className="text-foreground/80">{f.value}</span>
                    </div>
                  ))}
                </Card>
              </div>

              <div>
                <div className="font-semibold text-foreground font-display mb-3">Tags</div>
                <div className="flex flex-wrap gap-1.5">
                  {item.tags.map((t) => (
                    <span key={t} className="text-xs px-2 py-0.5 rounded bg-accent text-muted-foreground">{t}</span>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* files tab */}
          {detailTab === "files" && (
            <div className="animate-[fade-in_0.15s_ease]">
              <Card>
                <div className="px-4 py-2.5 border-b border-line flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-success">⊕</span>
                    <code className="text-foreground/80">{item.commits[0]?.hash}</code>
                    <span className="text-faint">{item.commits[0]?.message}</span>
                  </div>
                  <span className="text-faint">{item.files.length} files</span>
                </div>
                {item.files.map((f) => (
                  <div
                    key={f.name}
                    className="px-4 py-2.5 border-b border-input flex items-center justify-between cursor-pointer transition-colors hover:bg-input"
                  >
                    <div className="flex items-center gap-2">
                      <span style={{ color: FILE_TYPE_COLORS[f.type] ?? "var(--muted-foreground)" }}>
                        {(() => { const Icon = FILE_ICONS[f.type]; return Icon ? <Icon size={14} /> : <FileText size={14} />; })()}
                      </span>
                      <span className="text-foreground">{f.name}</span>
                    </div>
                    <span className="text-faint">{f.size}</span>
                  </div>
                ))}
              </Card>
            </div>
          )}

          {/* commits tab */}
          {detailTab === "commits" && (
            <div className="animate-[fade-in_0.15s_ease]">
              {item.commits.map((c, i) => (
                <div key={c.hash} className="flex items-start gap-3 py-3 border-b border-input">
                  <div className="flex flex-col items-center w-4 shrink-0 pt-1">
                    <div
                      className={cn(
                        "w-3 h-3 rounded-full border-2",
                        c.tag ? "bg-success border-success" : "bg-accent border-border",
                      )}
                    />
                    {i < item.commits.length - 1 && <div className="w-px flex-1 bg-border min-h-5 mt-1" />}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <code className="text-success font-medium">{c.hash}</code>
                      {c.tag && (
                        <span className="text-[8px] px-1.5 py-px rounded font-semibold border" style={{ background: "#22C55E12", color: "#22C55E", borderColor: "#22C55E20" }}>
                          {c.tag}
                        </span>
                      )}
                    </div>
                    <div className="text-foreground/80 mb-0.5">{c.message}</div>
                    <div className="text-faint">{c.author} · {c.date}</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* versions tab */}
          {detailTab === "versions" && (
            <div className="animate-[fade-in_0.15s_ease]">
              <div className="font-semibold text-foreground font-display mb-3">All Versions</div>
              {REPO_ITEMS.filter((r) => r.name === item.name && r.type === item.type).map((ver) => {
                const rtm = TYPE_META[ver.type];
                const isCurrent = ver.id === item.id;
                return (
                  <Card
                    key={ver.id}
                    className={cn("p-4 mb-2 cursor-pointer", isCurrent && "!bg-input")}
                    style={{ borderColor: isCurrent ? `${rtm.color}30` : undefined }}
                    onClick={() => navigate({ to: `/studio/hub/${ver.id}` })}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-2">
                        <code className={cn(isCurrent ? "text-foreground font-bold" : "text-subtle")}>{ver.version}</code>
                        {isCurrent && <span className="text-[8px] px-1.5 py-px rounded font-semibold" style={{ background: "#22C55E12", color: "#22C55E" }}>CURRENT</span>}
                        {ver.serving && <span className="text-[8px] px-1.5 py-px rounded font-semibold" style={{ background: "#22C55E12", color: "#22C55E" }}>SERVING</span>}
                      </div>
                      <span className="text-faint">{ver.updatedAt}</span>
                    </div>
                    <div className="text-subtle mb-1 truncate">{ver.description}</div>
                    <div className="flex gap-3 text-faint">
                      <span>{ver.size}</span>
                      <span className="flex items-center gap-1"><Download size={12} />{ver.downloads}</span>
                      <span className="flex items-center gap-1"><Heart size={12} />{ver.likes}</span>
                      <span>{ver.author}</span>
                    </div>
                  </Card>
                );
              })}
              {REPO_ITEMS.filter((r) => r.name === item.name && r.type === item.type).length <= 1 && (
                <div className="py-8 text-center text-faint">Only one version published</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
