/**
 * Finds the venv Python and runs generate_types.py.
 * Used by npm scripts and the Electron build pipeline.
 */
const { execFileSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const projectRoot = path.resolve(__dirname, "..", "..");
const isWindows = process.platform === "win32";

const venvPaths = isWindows
  ? [
      path.join(projectRoot, "venv", "Scripts", "python.exe"),
      path.join(projectRoot, ".venv", "Scripts", "python.exe"),
    ]
  : [
      path.join(projectRoot, "venv", "bin", "python"),
      path.join(projectRoot, ".venv", "bin", "python"),
    ];

const python =
  venvPaths.find((p) => fs.existsSync(p)) ||
  (isWindows ? "python" : "python3");

console.log(`[generate-types] Using Python: ${python}`);
console.log(`[generate-types] Project root: ${projectRoot}`);

const runModule = (mod, label) => {
  execFileSync(python, ["-m", mod], {
    cwd: projectRoot,
    stdio: "inherit",
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    // On Windows, shell: true is required for PATH resolution of Python
    // shims (.bat/.cmd, e.g. pyenv, conda). The arguments here are
    // hardcoded literals, not user input, so shell interpretation is not
    // a risk in this specific case. On non-Windows platforms, shell is
    // false, eliminating shell interpretation entirely.
    shell: isWindows,
  });
  console.log(`[${label}] Complete.`);
};

try {
  runModule("web.scripts.generate_types", "generate-types");
  runModule("web.scripts.generate_ui_schema", "generate-ui-schema");
} catch (err) {
  console.error("[generate-types] Failed:", err.message);
  process.exit(1);
}
