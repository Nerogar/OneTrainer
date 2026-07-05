import { Copy, X } from "lucide-react";
import { type ReactNode } from "react";

import { IconButton } from "./IconButton";

export interface ArrayItemHeaderProps {
  title: string;
  onClone: () => void;
  onRemove: () => void;
  /** Optional content before the title (e.g. an inline Toggle). */
  children?: ReactNode;
}

export function ArrayItemHeader({ title, onClone, onRemove, children }: ArrayItemHeaderProps) {
  return (
    <div className="flex justify-between items-start mb-4">
      <div className="flex items-center gap-3">
        {children}
        <h4 className="text-sm font-medium text-[var(--color-on-surface)]">{title}</h4>
      </div>
      <div className="flex gap-1">
        <IconButton
          icon={<Copy className="w-full h-full" />}
          label="Clone"
          variant="ghost"
          size="sm"
          onClick={onClone}
        />
        <IconButton
          icon={<X className="w-full h-full" />}
          label="Remove"
          variant="danger"
          size="sm"
          onClick={onRemove}
        />
      </div>
    </div>
  );
}
