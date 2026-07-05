import { type ButtonHTMLAttributes, forwardRef, type ReactNode } from "react";

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: ReactNode;
  label: string;
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
}

const sizeMap = { sm: "w-7 h-7", md: "w-9 h-9", lg: "w-11 h-11" };
const iconSizeMap = { sm: "w-3.5 h-3.5", md: "w-4 h-4", lg: "w-5 h-5" };

const variantClasses: Record<string, string> = {
  primary: "text-white",
  secondary: "bg-transparent border border-[var(--color-border-subtle)] text-[var(--color-on-surface)]",
  danger: "bg-[var(--color-error-500)] text-white",
  ghost: "bg-transparent text-[var(--color-on-surface-secondary)] hover:text-[var(--color-on-surface)]",
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ icon, label, variant = "ghost", size = "md", className = "", style, ...props }, ref) => {
    const isPrimary = variant === "primary";
    return (
      <button
        ref={ref}
        aria-label={label}
        className={`inline-flex items-center justify-center rounded-[var(--radius-sm)] cursor-pointer
          transition-all duration-200 ease-out hover:not-disabled:scale-103
          disabled:opacity-50 disabled:cursor-not-allowed
          ${sizeMap[size]} ${variantClasses[variant]} ${className}`}
        style={{
          ...(isPrimary
            ? { background: "linear-gradient(135deg, var(--color-cobalt-600), var(--color-azure-500))" }
            : {}),
          ...style,
        }}
        {...props}
      >
        <span className={iconSizeMap[size]}>{icon}</span>
      </button>
    );
  },
);
IconButton.displayName = "IconButton";
