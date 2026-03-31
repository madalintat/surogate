// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import MDEditor from "@uiw/react-md-editor";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Skill } from "@/types/skill";

/** Backend SkillStatus enum values. */
const SKILL_STATUSES = ["active", "error"] as const;

interface SkillFormProps {
  /** When provided the form is in edit mode, otherwise create mode. */
  skill?: Skill | null;
  onSave: (data: SkillFormData) => void;
  onCancel: () => void;
}

export interface SkillFormData {
  name: string;
  displayName: string;
  description: string;
  content: string;
  version: string;
  status: string;
  tags: string[];
}

export function SkillForm({ skill, onSave, onCancel }: SkillFormProps) {
  const isEdit = !!skill;

  const [name, setName] = useState(skill?.name ?? "");
  const [displayName, setDisplayName] = useState(skill?.displayName ?? "");
  const [description, setDescription] = useState(skill?.description ?? "");
  const [content, setContent] = useState(skill?.content ?? "");
  const [version, setVersion] = useState(skill?.version ?? "1.0.0");
  const [status, setStatus] = useState(skill?.status ?? "active");
  const [tagsRaw, setTagsRaw] = useState(skill?.tags.join(", ") ?? "");
  const [nameError, setNameError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!name.trim()) {
      setNameError("Name is required");
      return;
    }
    if (!/^[a-z0-9][a-z0-9\-]*$/.test(name.trim())) {
      setNameError("Lowercase alphanumeric and hyphens only, must start with a letter or digit");
      return;
    }
    setNameError(null);

    const tags = tagsRaw
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);

    onSave({
      name: name.trim(),
      displayName: displayName.trim() || name.trim(),
      description: description.trim(),
      content,
      version: version.trim() || "1.0.0",
      status,
      tags,
    });
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-150">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xl text-amber-500">&#x1F4C4;</span>
            <span className="text-base font-bold text-foreground font-display">
              {isEdit ? "Edit Skill" : "New Skill"}
            </span>
          </div>
          <Button variant="ghost" size="icon-xs" onClick={onCancel}>
            &#x2715;
          </Button>
        </div>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-6 space-y-4">
        {/* Name */}
        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">Name</label>
          <Input
            value={name}
            onChange={(e) => { setName(e.target.value); setNameError(null); }}
            placeholder="cx-support-skills"
            autoFocus
            className="font-mono"
          />
          {nameError
            ? <div className="text-destructive text-xs mt-1">{nameError}</div>
            : <div className="text-muted-foreground text-xs mt-1">Unique identifier (lowercase, hyphens allowed)</div>
          }
        </div>

        {/* Display Name */}
        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">Display Name</label>
          <Input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="CX Support Skills"
          />
        </div>

        {/* Description */}
        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">Description</label>
          <Textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Describe what this skill enables the agent to do..."
            rows={2}
          />
        </div>

        {/* Version & Status row */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">Version</label>
            <Input
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              placeholder="1.0.0"
              className="font-mono"
            />
          </div>
          <div>
            <label className="block mb-1 text-sm text-muted-foreground font-display">Status</label>
            <Select value={status} onValueChange={setStatus}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SKILL_STATUSES.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s.charAt(0).toUpperCase() + s.slice(1)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Tags */}
        <div>
          <label className="block mb-1 text-sm text-muted-foreground font-display">Tags</label>
          <Input
            value={tagsRaw}
            onChange={(e) => setTagsRaw(e.target.value)}
            placeholder="customer-support, empathy, escalation"
            className="font-mono"
          />
          <div className="text-muted-foreground text-xs mt-1">Comma-separated</div>
        </div>

        {/* Content */}
        <div data-color-mode="dark">
          <label className="block mb-1 text-sm text-muted-foreground font-display">Content</label>
          <MDEditor
            value={content}
            onChange={(v) => setContent(v ?? "")}
            height={320}
            preview="edit"
          />
          <div className="text-muted-foreground text-xs mt-1">Markdown skill definition</div>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-2 pt-2 pb-4">
          <Button variant="outline" type="button" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="submit" disabled={!name.trim()}>
            {isEdit ? "Save Changes" : "Create Skill"}
          </Button>
        </div>
      </form>
    </div>
  );
}
