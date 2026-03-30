// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect } from "react";
import { ArrowLeft, FileText, AlertTriangle } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { getObjectContent } from "@/api/hub";
import type { ObjectStats } from "@/types/hub";

const MAX_PREVIEW_SIZE = 5 * 1024 * 1024; // 5 MB

function humanSize(bytes: number): string {
  if (!bytes) return "0 B";
  const e = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, e)).toFixed(1) + " " + " KMGTP".charAt(e) + "B";
}

function getExtension(path: string): string {
  const dot = path.lastIndexOf(".");
  return dot >= 0 ? path.slice(dot + 1).toLowerCase() : "";
}

function getFileName(path: string): string {
  return path.split("/").pop() ?? path;
}

// ---- Sub-renderers ----

function JsonRenderer({ content }: { content: string }) {
  let formatted: string;
  try {
    formatted = JSON.stringify(JSON.parse(content), null, 2);
  } catch {
    formatted = content;
  }

  return (
    <pre className="p-4 text-sm font-mono text-foreground whitespace-pre-wrap break-words overflow-auto max-h-[600px]">
      {formatted}
    </pre>
  );
}

function MarkdownRenderer({ content }: { content: string }) {
  return (
    <MarkdownPreview
      markdown={content}
      className="!max-h-none !text-sm !p-5 !leading-normal"
    />
  );
}

function PlainTextRenderer({ content }: { content: string }) {
  return (
    <pre className="p-4 text-sm font-mono text-foreground whitespace-pre-wrap break-words overflow-auto max-h-[600px]">
      {content}
    </pre>
  );
}

function TooLargeRenderer({ object }: { object: ObjectStats }) {
  return (
    <div className="p-8 flex flex-col items-center gap-3 text-center">
      <AlertTriangle size={32} className="text-muted-foreground" />
      <div className="text-foreground font-display font-semibold">File too large to preview</div>
      <div className="text-faint text-sm">
        {getFileName(object.path)} is {humanSize(object.size_bytes ?? 0)}, which exceeds the {humanSize(MAX_PREVIEW_SIZE)} preview limit.
      </div>
    </div>
  );
}

function ContentRenderer({ object, content }: { object: ObjectStats; content: string }) {
  const ext = getExtension(object.path);

  switch (ext) {
    case "json":
      return <JsonRenderer content={content} />;
    case "md":
    case "markdown":
      return <MarkdownRenderer content={content} />;
    case "txt":
    case "log":
    case "csv":
    case "tsv":
    case "yaml":
    case "yml":
    case "toml":
    case "cfg":
    case "ini":
    case "py":
    case "js":
    case "ts":
    case "jsx":
    case "tsx":
    case "sh":
    case "bash":
    case "xml":
    case "html":
    case "css":
    case "sql":
    case "r":
    case "rs":
    case "go":
    case "java":
    case "c":
    case "cpp":
    case "h":
    case "hpp":
      return <PlainTextRenderer content={content} />;
    default:
      return <PlainTextRenderer content={content} />;
  }
}

// ---- Main component ----

interface ObjectRendererProps {
  repoId: string;
  activeRef: string;
  object: ObjectStats;
  onBack: () => void;
}

export function ObjectRenderer({ repoId, activeRef, object, onBack }: ObjectRendererProps) {
  const [content, setContent] = useState<string | null | undefined>(undefined);

  const tooLarge = (object.size_bytes ?? 0) > MAX_PREVIEW_SIZE;

  useEffect(() => {
    if (tooLarge) {
      setContent(null);
      return;
    }
    let cancelled = false;
    setContent(undefined); // eslint-disable-line react-hooks/set-state-in-effect
    getObjectContent(repoId, activeRef, object.path)
      .then((c) => { if (!cancelled) setContent(c); })
      .catch(() => { if (!cancelled) setContent(null); });
    return () => { cancelled = true; };
  }, [repoId, activeRef, object.path, tooLarge]);

  return (
    <Card>
      {/* header */}
      <div className="px-4 py-2.5 border-b border-line flex items-center gap-2">
        <button
          type="button"
          onClick={onBack}
          className="text-success cursor-pointer bg-transparent border-none hover:underline flex items-center gap-1"
        >
          <ArrowLeft size={14} />
          Back
        </button>
        <span className="text-faint">/</span>
        <FileText size={14} className="text-muted-foreground" />
        <span className="text-foreground font-mono text-sm">{object.path}</span>
        {object.size_bytes != null && (
          <span className="text-faint text-sm ml-auto">{humanSize(object.size_bytes)}</span>
        )}
      </div>

      {/* content */}
      {tooLarge && <TooLargeRenderer object={object} />}

      {!tooLarge && content === undefined && (
        <div className="p-4 space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}

      {!tooLarge && content === null && (
        <div className="p-8 text-center text-faint">Unable to load file content</div>
      )}

      {!tooLarge && content != null && (
        <ContentRenderer object={object} content={content} />
      )}
    </Card>
  );
}
