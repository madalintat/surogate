import type { Model } from "@/types/model";

export function isProxyModel(model: Model): boolean {
  return model.engine === "openrouter" || model.engine === "openai_compat";
}