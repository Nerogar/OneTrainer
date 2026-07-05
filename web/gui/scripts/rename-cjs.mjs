// Renames tsc's .js output to .cjs so Node loads it as CommonJS — package.json
// sets "type": "module", which would otherwise treat .js as ESM and refuse the
// emitted require()/exports syntax. Shared files are duplicated (.js kept for
// any future ESM consumer, .cjs produced for the main/preload loader).
import { copyFileSync, readdirSync, readFileSync, renameSync, writeFileSync } from "fs";
import { join } from "path";

const distMain = new URL("../dist/main", import.meta.url).pathname.replace(/^\/([A-Z]:)/, "$1");

function walk(dir) {
  const entries = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) entries.push(...walk(full));
    else entries.push(full);
  }
  return entries;
}

const allJsFiles = walk(distMain).filter((f) => f.endsWith(".js"));
const sharedDir = join(distMain, "shared");

const mainFiles = [];
const sharedFiles = [];

for (const f of allJsFiles) {
  if (f.startsWith(sharedDir)) sharedFiles.push(f);
  else mainFiles.push(f);
}

for (const file of mainFiles) {
  let content = readFileSync(file, "utf8");
  content = content.replace(/require\("(\.[^"]+?)"\)/g, (match, p1) => {
    if (p1.endsWith(".json") || p1.endsWith(".html") || p1.endsWith(".node")) return match;
    return `require("${p1}.cjs")`;
  });
  writeFileSync(file, content, "utf8");
}

for (const file of sharedFiles) {
  copyFileSync(file, file.replace(/\.js$/, ".cjs"));
}

for (const file of mainFiles) {
  renameSync(file, file.replace(/\.js$/, ".cjs"));
}

console.log(`Main: ${mainFiles.length} renamed to .cjs | Shared: ${sharedFiles.length} duplicated (.js + .cjs)`);
