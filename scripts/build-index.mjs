#!/usr/bin/env node
// Scans journal/ for .md / .markdown / .txt files and writes journal/index.json.
// Runs automatically in GitHub Actions on every push; run locally to preview.
// Usage: node scripts/build-index.mjs

import { readdir, readFile, writeFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DIR = path.join(ROOT, "journal");
const EXTS = new Set([".md", ".markdown", ".txt"]);

// Optional overrides + series definitions (content/journal.json).
const CONFIG_FILE = path.join(ROOT, "content", "journal.json");
const config = existsSync(CONFIG_FILE) ? JSON.parse(await readFile(CONFIG_FILE, "utf8")) : {};
const slugify = (s) => String(s).toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-|-$/g, "");

function parseFrontMatter(src) {
  const m = src.match(/^﻿?---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { meta: {}, body: src };
  const meta = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = line.match(/^([A-Za-z][\w-]*)\s*:\s*(.*)$/);
    if (!kv) continue;
    let v = kv[2].trim();
    if (/^\[.*\]$/.test(v)) {
      v = v.slice(1, -1).split(",").map((s) => s.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
    } else {
      v = v.replace(/^["']|["']$/g, "");
    }
    meta[kv[1].toLowerCase()] = v;
  }
  return { meta, body: src.slice(m[0].length) };
}

const humanize = (s) => s.replace(/^\d{4}-\d{2}-\d{2}[-_]?/, "").replace(/[-_]+/g, " ").replace(/^\w/, (c) => c.toUpperCase());

function firstParagraph(body, isMd) {
  const blocks = body.split(/\r?\n\s*\r?\n/).map((b) => b.trim()).filter(Boolean);
  const para = blocks.find((b) => !isMd || !/^(#|```|~~~|\||>|\$\$|!\[|---|[-*+] |\d+\. )/.test(b)) || "";
  const text = para.replace(/^\W*(TL;DR|Summary|Abstract)\W*/i, "").replace(/[*_`#>\[\]]|\(https?:[^)]*\)/g, "").replace(/\s+/g, " ").trim();
  return text.length > 180 ? text.slice(0, 177).replace(/\s+\S*$/, "") + "…" : text;
}

const files = (await readdir(DIR)).filter((f) => EXTS.has(path.extname(f).toLowerCase()) && !f.startsWith("_"));
const entries = [];

for (const file of files) {
  const raw = await readFile(path.join(DIR, file), "utf8");
  const { meta, body } = parseFrontMatter(raw);
  const ext = path.extname(file).toLowerCase();
  const isMd = ext !== ".txt";
  const base = path.basename(file, path.extname(file));
  const slug = slugify(base);
  for (const [k, v] of Object.entries(config.entries?.[slug] || config.entries?.[base] || {})) meta[k.toLowerCase()] = v;
  if (String(meta.draft).toLowerCase() === "true") continue;

  const h1 = isMd ? body.match(/^\s*#\s+(.+)$/m)?.[1] : null;
  const firstLine = !isMd ? body.trim().split(/\r?\n/)[0] : null;

  const sortKey = meta.date ? String(meta.date) : "";
  const date =
    (meta.date && String(meta.date).slice(0, 10)) ||
    base.match(/^(\d{4}-\d{2}-\d{2})/)?.[1] ||
    (await stat(path.join(DIR, file))).mtime.toISOString().slice(0, 10);

  const tags = Array.isArray(meta.tags) ? meta.tags : meta.tags ? String(meta.tags).split(",").map((t) => t.trim()) : [];

  entries.push({
    slug,
    file,
    type: isMd ? "md" : "txt",
    title: meta.title || h1?.trim() || (firstLine && firstLine.length < 90 ? firstLine : humanize(base)),
    date,
    tags: tags.filter(Boolean),
    summary: meta.summary || firstParagraph(isMd && h1 ? body.replace(/^\s*#\s+.+$/m, "") : body, isMd),
    words: body.split(/\s+/).filter(Boolean).length,
    _sort: sortKey || date,
    _series: meta.series ? String(meta.series) : null,
    _episode: Number(meta.episode) || null,
  });
}

// Number entries chronologically (No. 001 = oldest), then list newest first.
entries.sort((a, b) => a._sort.localeCompare(b._sort) || a.file.localeCompare(b.file));
entries.forEach((e, i) => (e.no = i + 1));

// Series: from content/journal.json, plus any post with `series: Name` (and optional `episode: N`) front matter.
const bySlug = Object.fromEntries(entries.map((e) => [e.slug, e]));
const series = (config.series || []).map((s) => ({
  slug: s.slug || slugify(s.title),
  title: s.title,
  description: s.description || "",
  entries: (s.entries || []).map(slugify).filter((x) => bySlug[x]),
}));
for (const e of entries) {
  if (!e._series || series.some((s) => s.entries.includes(e.slug))) continue;
  const key = slugify(e._series);
  let s = series.find((x) => x.slug === key);
  if (!s) series.push((s = { slug: key, title: e._series, description: "", entries: [] }));
  s.entries.push(e.slug);
}
for (const s of series) {
  if (s.entries.every((x) => bySlug[x]._episode)) s.entries.sort((a, b) => bySlug[a]._episode - bySlug[b]._episode);
  s.entries.forEach((x, i) => (bySlug[x].series = { slug: s.slug, title: s.title, part: i + 1, of: s.entries.length }));
}
for (const e of entries) for (const k of ["_sort", "_series", "_episode"]) delete e[k];
entries.reverse();

const index = { generated: new Date().toISOString(), series: series.filter((s) => s.entries.length), entries };
await writeFile(path.join(DIR, "index.json"), JSON.stringify(index, null, 2) + "\n");
console.log(`journal/index.json — ${entries.length} entr${entries.length === 1 ? "y" : "ies"}`);
