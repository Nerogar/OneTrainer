import { type ReactNode } from "react";

import { Card, type CardProps } from "./Card";

export interface SectionCardProps {
  title: string;
  children: ReactNode;
  className?: string;
  padding?: CardProps["padding"];
}

export function SectionCard({ title, children, className, padding }: SectionCardProps) {
  return (
    <Card className={`relative overflow-hidden pl-6 ${className ?? ""}`} padding={padding}>
      <div
        className="absolute left-0 top-0 bottom-0 w-[3px] rounded-full"
        style={{ background: "linear-gradient(180deg, var(--color-cobalt-600), var(--color-azure-500))" }}
      />
      <h3 className="text-sm font-semibold uppercase tracking-wider mb-4 text-[var(--color-on-surface-secondary)]">
        {title}
      </h3>
      {children}
    </Card>
  );
}
