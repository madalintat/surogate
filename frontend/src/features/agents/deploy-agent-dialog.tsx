import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { useTheme } from "@/hooks/use-theme";
import { useAppStore } from "@/stores/app-store";

// ── Deploy modal ───────────────────────────────────────────────

const DEPLOY_TEMPLATES = [
  { iconDark: "/hermes-agent-logo.png", iconLight: "/hermes-agent-logo.png", label: "Hermes Agent", desc: "Agent harness from NousResearch", harness: "hermes" },
  { iconDark: "/nanobot-logo.webp", iconLight: "/nanobot-logo.webp", label: "Nanobot ", desc: "Nanobot agent harness", harness: "nanobot" },
  { iconDark: "/openclaw-logo-dark.avif", iconLight: "/openclaw-logo-light.avif", label: "OpenClaw Agent", desc: "OpenClaw agent harness", harness: "openclaw" },
];

type Step = "harness" | "details";

export function DeployAgentDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { isDark } = useTheme();
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const createAgent = useAppStore((s) => s.createAgent);
  const models = useAppStore((s) => s.models);
  const fetchModels = useAppStore((s) => s.fetchModels);

  useEffect(() => {
    if (open && models.length === 0) fetchModels();
  }, [open, models.length, fetchModels]);

  const [step, setStep] = useState<Step>("harness");
  const [selectedHarness, setSelectedHarness] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState("");
  const name = displayName
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  const [description, setDescription] = useState("");
  const [modelId, setModelId] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const reset = () => {
    setStep("harness");
    setSelectedHarness(null);
    setDisplayName("");
    setDescription("");
    setModelId("");
    setSubmitting(false);
  };

  const handleClose = (v: boolean) => {
    if (!v) reset();
    onOpenChange(v);
  };

  const handleSelectHarness = (harness: string) => {
    setSelectedHarness(harness);
    setStep("details");
  };

  const handleCreate = async () => {
    if (!activeProjectId || !selectedHarness || !name || !modelId) return;
    setSubmitting(true);
    const agent = await createAgent(activeProjectId, {
      name,
      harness: selectedHarness,
      display_name: displayName || name,
      description,
      model_id: modelId,
    });
    setSubmitting(false);
    if (agent) {
      handleClose(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="min-w-150">
        <DialogHeader>
          <DialogTitle>Deploy New Agent</DialogTitle>
          <DialogDescription>
            {step === "harness" ? "Choose an agent harness" : "Configure your agent"}
          </DialogDescription>
        </DialogHeader>

        {step === "harness" && (
          <div className="grid grid-cols-3 gap-2.5">
            {DEPLOY_TEMPLATES.map((t) => (
              <button type="button"
                key={t.label}
                onClick={() => handleSelectHarness(t.harness)}
                className="p-3.5 rounded-lg border border-border bg-muted/40 hover:border-amber-500/30 hover:bg-muted/60 transition-colors cursor-pointer flex flex-col items-center justify-start gap-2.5"
              >
                <span className="text-lg text-amber-500 shrink-0">
                  <img src={isDark ? t.iconDark : t.iconLight} alt={t.label} className="h-8" />
                </span>
                <div>
                  <div className="font-semibold mt-1 mb-3">
                    {t.label}
                  </div>
                  <div className="text-muted-foreground mt-0.5">
                    {t.desc}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}

        {step === "details" && (
          <div className="space-y-3">
            <div>
              <label htmlFor="agent-display-name" className="text-xs font-medium text-muted-foreground mb-1 block">Display Name</label>
              <Input
                id="agent-display-name"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="My Agent"
                className="h-8 text-xs"
              />
            </div>
            <div>
              <label htmlFor="agent-name" className="text-xs font-medium text-muted-foreground mb-1 block">Identifier</label>
              <Input
                id="agent-name"
                value={name}
                readOnly
                placeholder="my-agent"
                className="h-8 text-xs bg-muted/40 text-muted-foreground"
              />
            </div>
            <div>
              <label htmlFor="agent-description" className="text-xs font-medium text-muted-foreground mb-1 block">Description</label>
              <Input
                id="agent-description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="What does this agent do?"
                className="h-8 text-xs"
              />
            </div>
            <div>
              <span className="text-xs font-medium text-muted-foreground mb-1 block">Model</span>
              <Select value={modelId} onValueChange={setModelId}>
                <SelectTrigger size="sm" className="w-full text-xs">
                  <SelectValue placeholder="Select a model..." />
                </SelectTrigger>
                <SelectContent>
                  {models.map((m) => (
                    <SelectItem key={m.id} value={m.id}>
                      {m.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        )}

        <DialogFooter>
          {step === "details" && (
            <Button variant="outline" onClick={() => setStep("harness")}>
              Back
            </Button>
          )}
          <Button variant="outline" onClick={() => handleClose(false)}>
            Cancel
          </Button>
          {step === "details" && (
            <Button
              onClick={handleCreate}
              disabled={!displayName.trim() || !modelId || submitting}
            >
              {submitting ? "Creating..." : "Create Agent"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
