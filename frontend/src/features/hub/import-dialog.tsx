// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect, useRef } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/utils/cn";
import { hfSearch, type HFItem } from "@/utils/hf";
import { Loader2, Download } from "lucide-react";

type ImportSource = "huggingface" | "modelscope";
type ImportType = "model" | "dataset";

interface ImportDialogProps {
  open: boolean;
  onClose: () => void;
  onImport: (params: {
    source: ImportSource;
    type: ImportType;
    repoId: string;
    subset?: string;
    token?: string;
  }) => void;
}

function ImportForm({ source, onImport, onClose }: { source: ImportSource; onImport: ImportDialogProps["onImport"]; onClose: () => void }) {
  const [type, setType] = useState<ImportType>("model");
  const [repoId, setRepoId] = useState("");
  const [subset, setSubset] = useState("");
  const [token, setToken] = useState("");

  // HuggingFace live search
  const [hfResults, setHfResults] = useState<HFItem[]>([]);
  const [hfLoading, setHfLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const justSelected = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (source !== "huggingface" || repoId.length < 3 || justSelected.current) {
      justSelected.current = false;
      setHfResults([]);
      setShowResults(false);
      return;
    }

    setHfLoading(true);
    const timer = setTimeout(async () => {
      try {
        const kind = type === "model" ? "models" : "datasets";
        const results = await hfSearch(kind, repoId, 10, token || undefined);
        setHfResults(results);
        setShowResults(results.length > 0);
      } catch {
        setHfResults([]);
        setShowResults(false);
      } finally {
        setHfLoading(false);
      }
    }, 300);

    return () => { clearTimeout(timer); setHfLoading(false); };
  }, [source, repoId, type, token]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!showResults) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowResults(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showResults]);

  const label = source === "huggingface" ? "HuggingFace" : "ModelScope";
  const placeholder = source === "huggingface" ? "meta-llama/Llama-3.1-8B" : "ZhipuAI/glm-4-9b";
  const tokenPlaceholder = source === "huggingface" ? "hf_..." : "ms_...";

  return (
    <>
      <div className="space-y-4">
        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">Type</label>
          <div className="flex gap-2">
            {(["model", "dataset"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setType(t)}
                className={cn(
                  "flex-1 px-3 py-1.5 rounded-md border cursor-pointer font-display capitalize transition-all",
                  type === t
                    ? "border-success/30 bg-success/10 text-success font-semibold"
                    : "border-border text-muted-foreground",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        <div ref={containerRef} className="relative">
          <label className="block mb-1 text-sm text-muted-foreground font-display">
            {label} Repository ID
          </label>
          <div className="relative">
            <Input
              value={repoId}
              onChange={(e) => setRepoId(e.target.value)}
              onFocus={() => { if (hfResults.length > 0) setShowResults(true); }}
              placeholder={placeholder}
              className="font-mono"
            />
            {hfLoading && (
              <Loader2 className="absolute right-2 top-1/2 -translate-y-1/2 animate-spin text-muted-foreground" size={14} />
            )}
          </div>
          {showResults && hfResults.length > 0 && (
            <div className="absolute z-50 mt-1 w-full max-h-56 overflow-y-auto rounded-lg border border-border bg-popover shadow-md">
              {hfResults.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="flex w-full items-center gap-2.5 px-2.5 py-2 text-left text-sm hover:bg-accent transition-colors cursor-pointer border-none bg-transparent"
                  onClick={() => {
                    justSelected.current = true;
                    setRepoId(item.id);
                    setShowResults(false);
                  }}
                >
                  <div className="flex-1 min-w-0">
                    <span className="font-mono text-foreground truncate block">{item.id}</span>
                    {item.author && (
                      <span className="text-xs text-muted-foreground">{item.author}</span>
                    )}
                  </div>
                  {item.downloads != null && item.downloads > 0 && (
                    <span className="flex items-center gap-1 text-xs text-faint shrink-0">
                      <Download size={10} />
                      {item.downloads.toLocaleString()}
                    </span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {type === "dataset" && (
          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">
              Subset <span className="text-faint">(optional)</span>
            </label>
            <Input
              value={subset}
              onChange={(e) => setSubset(e.target.value)}
              placeholder="default"
              className="font-mono"
            />
          </div>
        )}

        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">
            Access Token <span className="text-faint">(optional)</span>
          </label>
          <Input
            value={token}
            onChange={(e) => setToken(e.target.value)}
            type="password"
            placeholder={tokenPlaceholder}
            className="font-mono"
          />
        </div>
      </div>

      <DialogFooter>
        <Button variant="outline" onClick={onClose}>
          Cancel
        </Button>
        <Button
          disabled={!repoId.trim()}
          onClick={() => onImport({
            source,
            type,
            repoId: repoId.trim(),
            subset: subset.trim() || undefined,
            token: token.trim() || undefined,
          })}
        >
          Import
        </Button>
      </DialogFooter>
    </>
  );
}

export function ImportDialog({ open, onClose, onImport }: ImportDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onClose(); }}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Import Repository</DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="huggingface">
          <TabsList variant="line" className="w-full">
            <TabsTrigger value="huggingface">HuggingFace</TabsTrigger>
            <TabsTrigger value="modelscope">ModelScope</TabsTrigger>
          </TabsList>
          <TabsContent value="huggingface">
            <ImportForm source="huggingface" onImport={onImport} onClose={onClose} />
          </TabsContent>
          <TabsContent value="modelscope">
            <ImportForm source="modelscope" onImport={onImport} onClose={onClose} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
