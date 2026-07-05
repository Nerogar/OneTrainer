import "./styles/globals.css";
import "./styles/app.css";

import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { applyTheme, getInitialTheme } from "./styles/theme";

applyTheme(getInitialTheme());

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");

const params = new URLSearchParams(window.location.search);
const route = params.get("route");

if (route === "mask-editor") {
  const MaskEditorPage = lazy(() => import("./pages/MaskEditorPage"));
  const initialFolder = params.get("folder") ?? undefined;

  createRoot(root).render(
    <StrictMode>
      <Suspense fallback={<div className="flex items-center justify-center h-screen">Loading editor...</div>}>
        <MaskEditorPage initialFolder={initialFolder} />
      </Suspense>
    </StrictMode>,
  );
} else {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
