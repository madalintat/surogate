// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { RepositoryType } from "@/types/hub";
import type { RepositoryType as RepositoryTypeValue } from "@/types/hub";
import type { RepositoryCreation } from "@/types/hub";

interface CreateRepoDialogProps {
  open: boolean;
  error: string | null;
  onClose: () => void;
  onCreate: (req: RepositoryCreation) => Promise<boolean>;
}

export function CreateRepoDialog({ open, error, onClose, onCreate }: CreateRepoDialogProps) {
  const [name, setName] = useState("");
  const [nameError, setNameError] = useState<string | null>(null);
  const [type, setType] = useState<RepositoryTypeValue>(RepositoryType.MODEL);
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [creating, setCreating] = useState(false);

  const reset = () => {
    setName("");
    setNameError(null);
    setType(RepositoryType.MODEL);
    setDescription("");
    setTags("");
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!name.trim() || creating) return;
    if (!/^[a-zA-Z0-9][a-zA-Z0-9\-.]{2,62}$/.test(name.trim())) {
      setNameError("3–63 chars, alphanumeric start, only letters, digits, hyphens and dots");
      return;
    }
    setNameError(null);
    setCreating(true);
    const tagList = tags.split(",").map((t) => t.trim()).filter(Boolean);
    const metadata: Record<string, string> = { type };
    if (description.trim()) metadata.description = description.trim();
    if (tagList.length > 0) metadata.tags = tagList.join(",");
    const ok = await onCreate({
      name: name.trim(),
      storage_namespace: `local://${name.trim()}`,
      metadata,
    });
    setCreating(false);
    if (ok) handleClose();
  };

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) handleClose(); }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>New Repository</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">Repository name</label>
            <Input
              value={name}
              onChange={(e) => { setName(e.target.value); setNameError(null); }}
              placeholder="my-model"
              autoFocus
              className="font-mono"
            />
            {nameError
              ? <div className="text-destructive text-xs mt-1">{nameError}</div>
              : <div className="text-muted-foreground text-xs mt-1">3–63 chars, starts with a letter or digit</div>
            }
          </div>

          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">Type</label>
            <Select value={type} onValueChange={(v) => setType(v as RepositoryTypeValue)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Object.entries(RepositoryType).map(([key, value]) => (
                  <SelectItem key={value} value={value}>
                    {key.charAt(0) + key.slice(1).toLowerCase()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">Description</label>
            <Input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="A short description of this repository"
              className="font-mono"
            />
          </div>

          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">Tags</label>
            <Input
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="llama, fine-tuned, production"
              className="font-mono"
            />
          </div>

          {error && <div className="text-destructive text-sm">{error}</div>}

          <DialogFooter>
            <Button variant="outline" type="button" onClick={handleClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim() || creating}>
              {creating ? "Creating..." : "Create"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
