import { type ReactNode } from "react";

export interface CardProps {
  children: ReactNode;
  className?: string;
  hoverable?: boolean;
  padding?: "none" | "sm" | "md" | "lg";
}

const paddingClasses = { none: "", sm: "p-4", md: "p-5", lg: "p-8" };

export function Card({ children, className = "", hoverable = true, padding = "md" }: CardProps) {
  return (
    <div className={`card ${paddingClasses[padding]} ${hoverable ? "" : "card-static"} ${className}`}>{children}</div>
  );
}
