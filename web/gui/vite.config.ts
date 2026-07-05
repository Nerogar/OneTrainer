import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { rmSync, writeFileSync } from "fs";
import { resolve } from "path";
import { defineConfig, type Plugin } from "vite";

const PORT_FILE = resolve(__dirname, ".vite-port");

// Vite auto-increments past `server.port` when it's taken (strictPort is not
// set), but Electron's main process is a separate process that needs to know
// which port it actually landed on. This plugin writes the resolved port to
// a file next to this config once the dev server starts listening, so
// Electron (and the dev launch script) can discover it instead of assuming
// the preferred port was available.
function writeResolvedPortPlugin(): Plugin {
  return {
    name: "write-resolved-port",
    configureServer(server) {
      // Clear any stale port file from a previous run *before* this server
      // finishes binding, so a poller watching for the file never reads a
      // leftover value from an instance that already exited.
      rmSync(PORT_FILE, { force: true });
      server.httpServer?.once("listening", () => {
        const address = server.httpServer?.address();
        if (address && typeof address === "object") {
          writeFileSync(PORT_FILE, String(address.port), "utf-8");
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), writeResolvedPortPlugin()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src/renderer"),
      "@shared": resolve(__dirname, "src/shared"),
    },
  },
  root: ".",
  base: "./",
  build: {
    outDir: "dist/renderer",
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
});
