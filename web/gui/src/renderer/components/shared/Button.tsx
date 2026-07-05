import { Loader2 } from "lucide-react";
import { type ButtonHTMLAttributes, forwardRef, type ReactNode } from "react";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  children: ReactNode;
}

const variantClasses: Record<string, string> = {
  primary: "text-white border-none",
  secondary: "bg-transparent border border-[var(--color-border-subtle)] text-[var(--color-on-surface)]",
  danger: "bg-[var(--color-error-500)] text-white border-none",
  ghost: "bg-transparent border-none text-[var(--color-on-surface-secondary)]",
};

const sizeClasses: Record<string, string> = {
  sm: "py-1 px-3 text-sm",
  md: "py-2 px-4 text-base",
  lg: "py-3 px-6 text-base",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = "primary", size = "md", loading, disabled, children, className = "", style, ...props }, ref) => {
    const isPrimary = variant === "primary";
    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={`inline-flex items-center justify-center gap-2 font-semibold rounded-[var(--radius-sm)] cursor-pointer
          transition-all duration-200 ease-out
          hover:not-disabled:scale-103
          disabled:opacity-50 disabled:cursor-not-allowed
          ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
        style={{
          ...(isPrimary
            ? { background: "linear-gradient(135deg, var(--color-cobalt-600), var(--color-azure-500))" }
            : {}),
          ...style,
        }}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {children}
      </button>
    );
  },
);
Button.displayName = "Button";
