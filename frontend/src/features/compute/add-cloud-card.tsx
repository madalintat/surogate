import { Card } from "@/components/ui/card";
import { PROVIDER_COLORS, SUPPORTED_PROVIDERS } from "./compute-data";
import { useNavigate } from "@tanstack/react-router";
import { useTheme } from "@/hooks/use-theme";

export function AddCloudCard() {
    const navigate = useNavigate();
    const { isDark } = useTheme();

    return (
        <Card size="sm" className="overflow-hidden">
        <div className="px-4 py-2.5 border-b border-line">
          <span className="text-sm font-semibold text-foreground font-display">Connect Cloud</span>
        </div>
        <div className="grid grid-cols-5 gap-0 divide-x divide-line">
          {SUPPORTED_PROVIDERS.map(p => {
            const logo = isDark ? p.logoDark : p.logoWhite;
            return (
              <button type="button"
                key={p.key}
                onClick={() => navigate({ to: "/studio/connect-cloud", search: { provider: p.key } })}
                className="flex flex-col items-center justify-center gap-3 px-4 py-3.5 text-left hover:bg-card-hover transition-colors cursor-pointer"
              >
                {logo
                  ? <img src={logo} alt={p.name} className="w-25 h-15 shrink-0 object-contain" />
                  : <span className="text-lg" style={{ color: PROVIDER_COLORS[p.key] }}>{"\u2601"}</span>
                }
                <div className="min-w-0 flex flex-col items-center justify-center">
                  <div className="truncate">{p.name}</div>
                </div>
              </button>
            );
          })}
        </div>
      </Card>
    )
}