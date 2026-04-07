// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAppStore } from "@/stores/app-store";
import { getModelLogs, getModelEvents } from "@/api/models";
import type { ModelEvent } from "@/api/models";
import { STATUS_COLORS } from "./compute-data";
import { X, Terminal, Clock, User, Folder, Cpu, MapPin, ChevronRight, Globe, Layers, Gauge, Server, CalendarClock } from "lucide-react";
import { statusForDot, InfoRow, EXTENDED_WORKLOAD_COLORS } from "./detail-shared";
import type { ExtendedWorkload } from "./detail-shared";

export function ModelDetail({ item, onClose }: { item: ExtendedWorkload; onClose: () => void }) {
  const [logs, setLogs] = useState<string[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [events, setEvents] = useState<ModelEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [rightTab, setRightTab] = useState<"logs" | "events">("logs");
  const stopModel = useAppStore((s) => s.stopModel);
  const startModel = useAppStore((s) => s.startModel);
  const restartModel = useAppStore((s) => s.restartModel);
  const dot = statusForDot(item.status);
  const isActive = dot === "running" || dot === "deploying";
  const model = item._model;

  useEffect(() => {
    let cancelled = false;
    const fetchLogs = async () => {
      setLogsLoading(true);
      try {
        const res = await getModelLogs(item.id, { tail: 200 });
        if (!cancelled) setLogs(res.lines);
      } catch {
        if (!cancelled) setLogs(["(could not fetch logs)"]);
      } finally {
        if (!cancelled) setLogsLoading(false);
      }
    };
    void fetchLogs();
    const interval = isActive ? setInterval(() => void fetchLogs(), 3000) : undefined;
    return () => { cancelled = true; clearInterval(interval); };
  }, [item.id, isActive]);

  useEffect(() => {
    let cancelled = false;
    const fetchEvents = async () => {
      setEventsLoading(true);
      try {
        const res = await getModelEvents(item.id, { limit: 200 });
        if (!cancelled) setEvents(res.events);
      } catch {
        if (!cancelled) setEvents([]);
      } finally {
        if (!cancelled) setEventsLoading(false);
      }
    };
    void fetchEvents();
    return () => { cancelled = true; };
  }, [item.id]);

  const color = EXTENDED_WORKLOAD_COLORS[item.type] ?? "#6B7280";

  return (
    <td colSpan={9} className="p-0">
      <div className="bg-muted/30 border-t border-line animate-in fade-in duration-150">
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <div className="flex items-center gap-2.5">
            <StatusDot status={statusForDot(item.status)} />
            <span className="text-sm font-bold text-foreground font-display">{item.name}</span>
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-medium uppercase border"
              style={{ background: color + "12", color, borderColor: color + "20" }}
            >
              {item.type}
            </span>
            <span className="text-[11px] font-medium" style={{ color: STATUS_COLORS[item.status] ?? "var(--muted-foreground)" }}>
              {item.status}
            </span>
            {item.method !== "\u2014" && <span className="text-[11px] text-faint">{"\u00b7"} {item.method}</span>}
          </div>
          <div className="flex items-center gap-2">
            {model && (dot === "running" || dot === "deploying") && (
              <Button
                variant="outline"
                size="xs"
                className="text-destructive border-destructive/30 hover:bg-destructive/10"
                onClick={(e) => { e.stopPropagation(); void stopModel(model.id); }}
              >
                Stop
              </Button>
            )}
            {model && dot === "stopped" && (
              <Button
                variant="outline"
                size="xs"
                onClick={(e) => { e.stopPropagation(); void startModel(model.id); }}
              >
                Start
              </Button>
            )}
            {model && dot === "error" && (
              <Button
                variant="outline"
                size="xs"
                onClick={(e) => { e.stopPropagation(); void restartModel(model.id); }}
              >
                Restart
              </Button>
            )}
            <Button variant="ghost" size="icon-xs" onClick={(e) => { e.stopPropagation(); onClose(); }}>
              <X size={14} />
            </Button>
          </div>
        </div>

        <div className="flex">
          {/* Left: info */}
          <div className="w-80 shrink-0 px-4 py-3 border-r border-line">
            <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
              <InfoRow icon={Folder} label="Project" value={item.project} />
              <InfoRow icon={User} label="Deployed by" value={item.requestedBy} />
              <InfoRow icon={MapPin} label="Location" value={item.location === "local" ? "Local Cluster" : `${item.location.charAt(0).toUpperCase() + item.location.slice(1)} Cloud`} />
              <InfoRow icon={Cpu} label="GPU" value={item.gpu} />
              {item.node !== "\u2014" && <InfoRow icon={ChevronRight} label="Node" value={item.node} />}
              {item.startedAt && <InfoRow icon={Clock} label="Started" value={item.startedAt} />}
              {model && <InfoRow icon={Server} label="Engine" value={model.engine} />}
              {model && <InfoRow icon={Layers} label="Replicas" value={`${model.replicas.current}/${model.replicas.desired}`} />}
              {model?.endpoint && model.endpoint !== "\u2014" && <InfoRow icon={Globe} label="Endpoint" value={model.endpoint} />}
              {model?.uptime && model.uptime !== "\u2014" && <InfoRow icon={Clock} label="Uptime" value={model.uptime} />}
              {model && model.tps > 0 && <InfoRow icon={Gauge} label="TPS" value={String(model.tps)} />}
            </div>
          </div>

          {/* Right: logs / events */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex items-center gap-3 px-3 py-1.5 border-b border-line shrink-0">
              <button
                type="button"
                onClick={() => setRightTab("logs")}
                className="flex items-center gap-1.5 cursor-pointer"
                style={{ opacity: rightTab === "logs" ? 1 : 0.45 }}
              >
                <Terminal size={11} className="text-faint" />
                <span className="text-[10px] font-semibold text-muted-foreground font-display uppercase tracking-wider">Logs</span>
                {isActive && rightTab === "logs" && <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />}
              </button>
              <button
                type="button"
                onClick={() => setRightTab("events")}
                className="flex items-center gap-1.5 cursor-pointer"
                style={{ opacity: rightTab === "events" ? 1 : 0.45 }}
              >
                <CalendarClock size={11} className="text-faint" />
                <span className="text-[10px] font-semibold text-muted-foreground font-display uppercase tracking-wider">Events</span>
                {events.length > 0 && (
                  <span className="text-[9px] px-1 rounded bg-muted text-faint">{events.length}</span>
                )}
              </button>
            </div>

            {rightTab === "logs" ? (
              <ScrollArea className="h-70 bg-background">
                <div className="p-3">
                  {logsLoading && logs.length === 0 ? (
                    <div className="text-[11px] text-faint animate-pulse">Loading logs...</div>
                  ) : logs.length === 0 ? (
                    <div className="text-[11px] text-faint">No log output yet</div>
                  ) : (
                    <pre className="text-[11px] leading-[1.6] text-muted-foreground font-mono whitespace-pre-wrap break-all">
                      {logs.map((line, i) => (
                        <div
                          key={i}
                          className={
                            line.startsWith("ERROR:") ? "text-destructive font-semibold" :
                            line.startsWith("Warning:") ? "text-[#F59E0B]" :
                            ""
                          }
                        >
                          {line}
                        </div>
                      ))}
                    </pre>
                  )}
                </div>
              </ScrollArea>
            ) : (
              <ScrollArea className="h-70 bg-background">
                <div className="p-3">
                  {eventsLoading && events.length === 0 ? (
                    <div className="text-[11px] text-faint animate-pulse">Loading events...</div>
                  ) : events.length === 0 ? (
                    <div className="text-[11px] text-faint">No events recorded</div>
                  ) : (
                    <div className="space-y-1.5">
                      {events.map((ev, i) => (
                        <div key={i} className="flex items-start gap-2 text-[11px]">
                          <span className="shrink-0 text-faint tabular-nums">
                            {new Date(ev.time).toLocaleString(undefined, {
                              month: "short", day: "numeric",
                              hour: "2-digit", minute: "2-digit", second: "2-digit",
                            })}
                          </span>
                          <span
                            className="shrink-0 text-[9px] px-1 py-0.5 rounded font-medium uppercase border"
                            style={{
                              color: ev.type === "job" ? "#3B82F6" : ev.type === "run" ? "#8B5CF6" : "var(--faint)",
                              background: ev.type === "job" ? "#3B82F610" : ev.type === "run" ? "#8B5CF610" : "var(--muted)",
                              borderColor: ev.type === "job" ? "#3B82F620" : ev.type === "run" ? "#8B5CF620" : "var(--border)",
                            }}
                          >
                            {ev.type}
                          </span>
                          <span className="text-muted-foreground">{ev.text}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </ScrollArea>
            )}
          </div>
        </div>
      </div>
    </td>
  );
}
