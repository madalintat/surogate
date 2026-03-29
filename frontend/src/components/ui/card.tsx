// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { cn } from "@/utils/cn";
import type { HTMLAttributes, ReactNode } from "react";

interface CardProps extends HTMLAttributes<HTMLElement> {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className, ...rest }: CardProps) {
  return (
    <section
      className={cn(
        "bg-muted border border-line rounded-lg overflow-hidden",
        className,
      )}
      {...rest}
    >
      {children}
    </section>
  );
}

interface CardHeaderProps {
  icon?: ReactNode;
  iconColor?: string;
  title: string;
  badge?: ReactNode;
  actions?: ReactNode;
}

export function CardHeader({
  icon,
  iconColor,
  title,
  badge,
  actions,
}: CardHeaderProps) {
  return (
    <div className="px-4 py-3 border-b border-line flex items-center justify-between">
      <div className="flex items-center gap-2">
        {icon && <span style={{ color: iconColor }}>{icon}</span>}
        <span className="font-display text-[13px] font-semibold text-foreground">
          {title}
        </span>
        {badge}
      </div>
      {actions}
    </div>
  );
}
