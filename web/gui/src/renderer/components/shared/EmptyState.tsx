import type { ReactNode } from "react";

import { Button } from "./Button";

export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 animate-[rowFade_0.3s_ease-out]">
      {icon && (
        <div className="text-[var(--color-on-surface-secondary)] opacity-30 mb-4 [&>svg]:w-12 [&>svg]:h-12">{icon}</div>
      )}
      <h3 className="text-lg font-medium text-[var(--color-on-surface)]">{title}</h3>
      {description && <p className="mt-1 text-sm text-[var(--color-on-surface-secondary)]">{description}</p>}
      {action && (
        <div className="mt-5">
          <Button variant="primary" size="sm" onClick={action.onClick}>
            {action.label}
          </Button>
        </div>
      )}
    </div>
  );
}
