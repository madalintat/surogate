// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useId } from "react";
import { Upload, FileText, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

function humanSize(bytes: number | undefined): string {
  if (!bytes) return "—";
  const e = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, e)).toFixed(1) + " " + " KMGTP".charAt(e) + "B";
}

interface UploadDialogProps {
  destinationPath: string;
  onUpload: (files: File[], path: string, onProgress: (name: string, percent: number) => void) => Promise<void>;
  onClose: () => void;
}

export function UploadDialog({ destinationPath, onUpload, onClose }: UploadDialogProps) {
  const inputId = useId();
  const [path, setPath] = useState(destinationPath);
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (files.length === 0 || uploading) return;
    setUploading(true);
    setError(null);
    try {
      await onUpload(files, path, (name, percent) => {
        setProgress((prev) => ({ ...prev, [name]: percent }));
      });
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <Card className="w-full max-w-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-foreground font-display">Upload Objects</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-muted-foreground cursor-pointer bg-transparent border-none"
          >
            <X size={16} />
          </button>
        </div>

        <div className="mb-3">
          <label className="block mb-1 text-sm text-muted-foreground font-display">Destination path</label>
          <Input
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="e.g. folder/subfolder/"
            className="font-mono"
          />
        </div>

        <div
          className="border-2 border-dashed border-border rounded-lg p-8 mb-4 text-center cursor-pointer hover:border-success/50 transition-colors"
          onClick={() => document.getElementById(inputId)?.click()}
          onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            const dropped = Array.from(e.dataTransfer.files);
            if (dropped.length) setFiles((prev) => [...prev, ...dropped]);
          }}
        >
          <Upload size={24} className="mx-auto mb-2 text-muted-foreground" />
          <p className="text-muted-foreground text-sm">Drag & drop files here, or click to select</p>
          <input
            id={inputId}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              const selected = Array.from(e.target.files ?? []);
              if (selected.length) setFiles((prev) => [...prev, ...selected]);
              e.target.value = "";
            }}
          />
        </div>

        {files.length > 0 && (
          <div className="mb-4 max-h-48 overflow-y-auto">
            <div className="text-sm font-display text-muted-foreground mb-2">
              {files.length} file{files.length > 1 ? "s" : ""} selected
            </div>
            {files.map((file, i) => (
              <div key={`${file.name}-${i}`} className="flex items-center gap-2 py-1 border-b border-input">
                <FileText size={12} className="text-muted-foreground shrink-0" />
                <span className="text-foreground text-sm flex-1 truncate">{file.name}</span>
                <span className="text-faint text-xs">{humanSize(file.size)}</span>
                {progress[file.name] != null && (
                  <div className="w-16 h-1.5 bg-input rounded-full overflow-hidden">
                    <div
                      className="h-full bg-success rounded-full transition-all"
                      style={{ width: `${progress[file.name]}%` }}
                    />
                  </div>
                )}
                {!uploading && (
                  <button
                    type="button"
                    onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                    className="text-muted-foreground cursor-pointer bg-transparent border-none"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {error && <div className="text-destructive text-sm mb-3">{error}</div>}

        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={uploading}
            className="px-3 py-1.5 rounded-md border border-border bg-input text-muted-foreground cursor-pointer font-display disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={files.length === 0 || uploading}
            className="px-3 py-1.5 rounded-md border border-success/30 bg-success/10 text-success cursor-pointer font-display disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>
      </Card>
    </div>
  );
}
