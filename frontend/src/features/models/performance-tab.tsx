// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState } from "react";
import { Sparkline } from "@/components/ui/sparkline";
import { Skeleton } from "@/components/ui/skeleton";
import type { Model } from "@/types/model";
import { getMetrics, getMetricsSummary, type MetricsBucket, type MetricsSummary } from "@/api/metrics";

type Period = "minute" | "hour" | "day";

const PERIODS: { id: Period; label: string; summaryHours: number }[] = [
  { id: "minute", label: "Last hour", summaryHours: 1 },
  { id: "hour", label: "Last 24h", summaryHours: 24 },
  { id: "day", label: "Last 30d", summaryHours: 720 },
];

export function PerformanceTab({ model }: { model: Model }) {
  const [period, setPeriod] = useState<Period>("hour");
  const [buckets, setBuckets] = useState<MetricsBucket[]>([]);
  const [summary, setSummary] = useState<MetricsSummary | null>(null);
  const [loading, setLoading] = useState(true);

  const activePeriod = PERIODS.find(p => p.id === period)!;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    Promise.all([
      getMetrics({ model: model.name, period }),
      getMetricsSummary({ model: model.name, hours: activePeriod.summaryHours }),
    ]).then(([metricsRes, summaryRes]) => {
      if (cancelled) return;
      setBuckets(metricsRes.buckets);
      setSummary(summaryRes);
      setLoading(false);
    }).catch(() => {
      if (!cancelled) setLoading(false);
    });

    return () => { cancelled = true; };
  }, [model.name, period, activePeriod.summaryHours]);

  const tpsData = buckets.map(b => b.tokens_per_sec);
  const latencyData = buckets.map(b => b.avg_latency_ms);
  const tokensData = buckets.map(b => b.total_tokens);
  const requestsData = buckets.map(b => b.request_count);

  const charts = [
    { label: "Throughput (tok/s)", data: tpsData, color: "#22C55E", unit: "tok/s", value: summary?.tokens_per_sec },
    { label: "Avg Latency (ms)", data: latencyData, color: "#3B82F6", unit: "ms", value: summary?.avg_latency_ms },
    { label: "Total Tokens", data: tokensData, color: "#8B5CF6", unit: "tok", value: summary?.total_tokens },
    { label: "Requests", data: requestsData, color: "#F59E0B", unit: "reqs", value: summary?.request_count },
  ];

  const successRate = summary ? (summary.success_rate * 100) : null;

  return (
    <div className="animate-in fade-in duration-150 space-y-4">
      {/* Period selector */}
      <div className="flex gap-1">
        {PERIODS.map(p => (
          <button
            key={p.id}
            onClick={() => setPeriod(p.id)}
            className={`px-2.5 py-1 rounded text-xs transition-colors cursor-pointer ${
              period === p.id
                ? "bg-primary/10 text-primary border border-primary/20"
                : "text-muted-foreground hover:bg-muted/50"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-5 gap-3">
        {[
          { label: "Tokens/sec", value: summary?.tokens_per_sec.toFixed(1), color: "#22C55E" },
          { label: "Avg Latency", value: summary ? `${summary.avg_latency_ms.toFixed(0)}ms` : null, color: "#3B82F6" },
          { label: "Total Tokens", value: summary?.total_tokens.toLocaleString(), color: "#8B5CF6" },
          { label: "Requests", value: summary?.request_count.toLocaleString(), color: "#F59E0B" },
          { label: "Success Rate", value: successRate != null ? `${successRate.toFixed(1)}%` : null, color: successRate != null && successRate < 99 ? "#EF4444" : "#22C55E" },
        ].map(kpi => (
          <div key={kpi.label} className="bg-muted/40 border border-border rounded-lg p-3">
            <div className="text-[10px] text-faint uppercase tracking-wider font-display mb-1">{kpi.label}</div>
            {loading ? (
              <Skeleton className="h-6 w-16 rounded" />
            ) : (
              <div className="text-lg font-bold" style={{ color: kpi.color }}>{kpi.value ?? "\u2014"}</div>
            )}
          </div>
        ))}
      </div>

      {/* Sparkline charts */}
      <div className="grid grid-cols-2 gap-4">
        {charts.map((chart) => (
          <section
            key={chart.label}
            className="bg-muted/40 border border-border rounded-lg overflow-hidden"
          >
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <span className="text-xs font-semibold text-foreground font-display">
                {chart.label}
              </span>
              <span className="text-[11px] font-semibold" style={{ color: chart.color }}>
                {loading ? "\u2014" : (chart.data.length > 0 ? chart.data[chart.data.length - 1].toFixed(1) : "\u2014")}{" "}
                <span className="text-[9px] text-muted-foreground font-normal">{chart.unit}</span>
              </span>
            </div>
            <div className="p-4 pb-3">
              {loading ? (
                <Skeleton className="h-20 w-full rounded" />
              ) : chart.data.length > 0 ? (
                <>
                  <Sparkline
                    data={chart.data.length === 1 ? [chart.data[0], chart.data[0]] : chart.data}
                    color={chart.color}
                    height={80}
                    width={420}
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground/40 mt-2 font-display">
                    <span>{activePeriod.label.replace("Last ", "")}{" ago"}</span>
                    <span>now</span>
                  </div>
                </>
              ) : (
                <div className="h-20 flex items-center justify-center text-[11px] text-faint">
                  No data for this period
                </div>
              )}
            </div>
          </section>
        ))}
      </div>

      {/* Token breakdown */}
      {summary && !loading && (
        <section className="bg-muted/40 border border-border rounded-lg p-4">
          <div className="text-xs font-semibold text-foreground font-display mb-3.5">
            Token Breakdown
          </div>
          <div className="flex gap-8 items-end">
            {[
              { label: "Prompt Tokens", value: summary.total_prompt_tokens.toLocaleString(), color: "#3B82F6" },
              { label: "Completion Tokens", value: summary.total_completion_tokens.toLocaleString(), color: "#22C55E" },
              { label: "Total Tokens", value: summary.total_tokens.toLocaleString(), color: "#8B5CF6" },
            ].map(s => (
              <div key={s.label} className="flex-1 flex flex-col items-center gap-1.5">
                <div className="text-base font-bold" style={{ color: s.color }}>{s.value}</div>
                <div className="text-[9px] text-muted-foreground font-display">{s.label}</div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
