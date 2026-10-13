#!/usr/bin/env node
// Mirrors posts from a local folder (default ~/Documents/Blogs) into journal/, then rebuilds the index.
// Your originals are never modified. Copies get a `date:` front-matter line (from the file's
// modified time) when they don't already have one, so dates survive the trip through GitHub.
// Usage: node scripts/sync-blogs.mjs [source-folder]

import { readdir, readFile, writeFile, stat, mkdir, copyFile, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEST = path.join(ROOT, "journal");
const SRC = path.resolve((process.argv[2] || "~/Documents/Blogs").replace(/^~(?=$|\/)/, os.homedir()));
const LEDGER = path.join(DEST, ".synced.json"); // which files in journal/ came from SRC
const POSTS = new Set([".md", ".markdown", ".txt"]);
const ASSETS = new Set([".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".pdf", ".csv"]);

if (!existsSync(SRC)) {
  console.error(`Source folder not found: ${SRC}`);
  process.exit(1);
}

const pad = (n) => String(n).padStart(2, "0");
const stamp = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;

function withDate(src, date) {
  const fm = src.match(/^﻿?---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (fm && /^date\s*:/m.test(fm[1])) return src;
  if (fm) return src.replace(/^﻿?---\r?\n/, `---\ndate: ${date}\n`);
  return `---\ndate: ${date}\n---\n${src}`;
}

// Walk SRC (skipping dot-folders like .git), collecting posts and their assets.
async function walk(dir, rel = "") {
  const out = [];
  for (const d of await readdir(dir, { withFileTypes: true })) {
    if (d.name.startsWith(".")) continue;
    const r = path.join(rel, d.name);
    if (d.isDirectory()) out.push(...(await walk(path.join(dir, d.name), r)));
    else out.push(r);
  }
  return out;
}

const previous = existsSync(LEDGER) ? JSON.parse(await readFile(LEDGER, "utf8")) : [];
const synced = [];
let posts = 0, assets = 0;

for (const rel of await walk(SRC)) {
  const ext = path.extname(rel).toLowerCase();
  const from = path.join(SRC, rel);
  const to = path.join(DEST, rel);

  if (POSTS.has(ext) && !rel.includes(path.sep)) {
    const { mtime } = await stat(from);
    await writeFile(to, withDate(await readFile(from, "utf8"), stamp(mtime)));
    posts++;
  } else if (ASSETS.has(ext)) {
    await mkdir(path.dirname(to), { recursive: true });
    await copyFile(from, to);
    assets++;
  } else continue;
  synced.push(rel);
}

// Remove files that were synced before but have since been deleted from SRC.
const removed = previous.filter((f) => !synced.includes(f));
for (const f of removed) await rm(path.join(DEST, f), { force: true });

await writeFile(LEDGER, JSON.stringify(synced.sort(), null, 2) + "\n");
console.log(`Synced ${posts} posts, ${assets} assets from ${SRC}${removed.length ? `, removed ${removed.length}` : ""}`);

execFileSync(process.execPath, [path.join(ROOT, "scripts", "build-index.mjs")], { stdio: "inherit" });
