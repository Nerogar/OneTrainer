export type Theme = "light" | "dark";

const STORAGE_KEY = "theme";

export function getInitialTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function applyTheme(theme: Theme): void {
  document.body.setAttribute("data-theme", theme);
  localStorage.setItem(STORAGE_KEY, theme);
}
