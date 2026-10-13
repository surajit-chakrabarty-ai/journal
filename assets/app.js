// Research Journal — tiny client-side renderer. No build step required.

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const pad = (n, w = 3) => String(n).padStart(w, "0");
const fmtDate = (d) =>
  new Date(d + "T00:00:00").toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
const entryUrl = (e) => `entry.html?e=${encodeURIComponent(e.slug)}`;

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

let profilePromise;
const loadProfile = () => (profilePromise ??= getJSON("content/profile.json"));

/* ───────── Page views (GoatCounter — free, cookie-less) ───────── */
const entryPath = (e) => `/journal/${e.slug}`;

async function trackView(path, title) {
  const code = (await loadProfile().catch(() => ({}))).analytics?.goatcounter;
  if (!code) return;
  window.goatcounter = { no_onload: true }; // we count explicit, clean paths ourselves
  const s = document.createElement("script");
  s.async = true;
  s.src = "https://gc.zgo.at/count.js";
  s.dataset.goatcounter = `https://${code}.goatcounter.com/count`;
  s.onload = () => window.goatcounter.count({ path, title: title || document.title });
  document.head.append(s);
}

async function viewCount(path) {
  const a = (await loadProfile().catch(() => ({}))).analytics || {};
  if (!a.goatcounter || a.showViews === false) return null;
  try {
    const res = await fetch(`https://${a.goatcounter}.goatcounter.com/counter/${encodeURIComponent(path)}.json`);
    if (res.status === 404) return "0"; // no hits yet
    return res.ok ? (await res.json()).count : null;
  } catch {
    return null;
  }
}

// Fill every [data-views] placeholder on the page with its count.
function fillViews(root = document) {
  root.querySelectorAll("[data-views]:not([data-filled])").forEach(async (el) => {
    el.dataset.filled = "1";
    const n = await viewCount(el.dataset.views);
    if (n == null) return;
    el.textContent = `${n} ${n === "1" ? "view" : "views"}`;
    el.hidden = false;
  });
}

const loadIndex = () => getJSON("journal/index.json").catch(() => ({ entries: [] }));

// Strip optional front matter (--- key: value ---) from a file.
function stripFrontMatter(src) {
  return src.replace(/^﻿?---\r?\n[\s\S]*?\r?\n---\r?\n?/, "");
}

// Markdown → HTML with $inline$ and $$display$$ math via KaTeX.
function renderMarkdown(src) {
  const code = [];
  const math = [];
  src = src.replace(/(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`)/g, (m) => `KTXCODE${code.push(m) - 1}KTX`);
  src = src
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => `KTXMATH${math.push([t, true]) - 1}KTX`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, t) => `KTXMATH${math.push([t, false]) - 1}KTX`)
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => `KTXMATH${math.push([t, true]) - 1}KTX`)
    .replace(/(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g, (_, pre, t) => `${pre}KTXMATH${math.push([t, false]) - 1}KTX`);
  src = src.replace(/KTXCODE(\d+)KTX/g, (_, i) => code[i]);

  let html = window.marked ? marked.parse(src, { gfm: true }) : `<pre class="plain">${esc(src)}</pre>`;
  html = html.replace(/(<img[^>]+src=")(?![a-z]+:|\/|#|journal\/)/gi, "$1journal/");
  return html.replace(/KTXMATH(\d+)KTX/g, (_, i) => {
    const [tex, display] = math[i];
    if (!window.katex) return esc(display ? `$$${tex}$$` : `$${tex}$`);
    return katex.renderToString(tex, { displayMode: display, throwOnError: false });
  });
}

function entryRow(e, withSummary = false) {
  const tags = (e.tags || []).length ? ` · ${e.tags.map(esc).join(", ")}` : "";
  return `<li><a href="${entryUrl(e)}">
    <span class="e-no">No. ${pad(e.no)}</span>
    <span class="e-title"><span>${esc(e.title)}</span></span>
    <time datetime="${e.date}">${fmtDate(e.date)}</time>
    ${withSummary ? `<span class="e-sum">${esc(e.summary || "")}${tags}<span class="e-views" data-views="${entryPath(e)}" hidden></span></span>` : ""}
  </a></li>`;
}

function showError(el, err) {
  const hint = location.protocol === "file:" ? " Serve the folder over HTTP (see README) instead of opening the file directly." : "";
  el.innerHTML = `<p class="empty">Couldn’t load content (${esc(err.message)}).${hint}</p>`;
}

/* ───────── Landing ───────── */
async function home() {
  const [p, idx] = await Promise.all([loadProfile(), loadIndex()]);
  const inline = (t) => (window.marked ? marked.parseInline(t) : esc(t));

  document.title = `${p.name} — Research Journal`;
  const words = String(p.name || "").trim().split(/\s+/);
  const last = words.length > 1 ? words.pop() : "";
  $("#name").innerHTML = last
    ? `<span>${esc(words.join(" "))}</span> <em>${esc(last)}<span class="dot">.</span></em>`
    : `${esc(p.name)}<span class="dot">.</span>`;
  $("#name").setAttribute("aria-label", p.name);
  if (p.photo) $("#photo").src = p.photo;
  $("#photo").alt = `Portrait of ${p.name}`;
  $("#role").textContent = p.role || "";
  $("#bio").innerHTML = (p.bio || []).map((t) => `<p>${inline(t)}</p>`).join("");
  $("#links").innerHTML = (p.links || [])
    .map((l) => `<li><a href="${esc(l.url)}" rel="noopener">${esc(l.label)}</a></li>`)
    .join("");

  const notes = [...(p.notes || [])];
  notes.push(["Entries", String(idx.entries.length)]);
  if (idx.entries[0]) notes.push(["Last entry", fmtDate(idx.entries[0].date)]);
  viewCount("TOTAL").then((n) => {
    if (n != null) $("#notes").insertAdjacentHTML("beforeend", `<dt>Total views</dt><dd>${esc(n)}</dd>`);
  });
  $("#notes").innerHTML = notes.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

  $("#projects").innerHTML = (p.projects || [])
    .map((pr, i) => {
      const title = pr.link ? `<a href="${esc(pr.link)}">${esc(pr.title)}</a>` : esc(pr.title);
      const status = (pr.status || "active").toLowerCase();
      const meta = [...(pr.tags || []), pr.since ? `since ${pr.since}` : ""].filter(Boolean).map(esc).join(" · ");
      return `<li class="project">
        <span class="p-num">${pad(i + 1, 2)}</span>
        <div>
          <h3>${title}<span class="status ${esc(status)}">${esc(status)}</span></h3>
          <p>${inline(pr.summary || "")}</p>
          ${meta ? `<p class="p-meta">${meta}</p>` : ""}
        </div>
      </li>`;
    })
    .join("");

  $("#recent").innerHTML = idx.entries.length
    ? idx.entries.slice(0, 4).map((e) => entryRow(e)).join("")
    : `<li class="empty">No entries yet — drop a .md or .txt file into /journal.</li>`;
}

/* ───────── Journal index ───────── */
const readTime = (e) => `${Math.max(1, Math.round(e.words / 220))} min read`;
// "LLM Inference Deep Dive: Episode 2 — The Optimization Playbook" → "The Optimization Playbook"
const shortTitle = (t) => t.split(/\s[—–]\s|:\s/).pop();
const monthLabel = (d) => new Date(d + "T00:00:00").toLocaleDateString("en-GB", { month: "long", year: "numeric" });
const dayLabel = (d) => new Date(d + "T00:00:00").toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
const seriesBadge = (e) =>
  e.series ? `<span class="s-badge">${esc(e.series.title)} · Part ${e.series.part}/${e.series.of}</span>` : "";

function timelineItem(e) {
  const tags = (e.tags || []).map((t) => `#${esc(t)}`).join(" ");
  return `<li class="tl-item${e.series ? " in-series" : ""}">
    <time class="tl-date" datetime="${e.date}">${dayLabel(e.date)}</time>
    <span class="tl-dot" aria-hidden="true"></span>
    <a class="tl-card" href="${entryUrl(e)}">
      ${seriesBadge(e)}
      <h3>${esc(e.title)}</h3>
      ${e.summary ? `<p>${esc(e.summary)}</p>` : ""}
      <span class="tl-meta">No. ${pad(e.no)} · ${readTime(e)}${tags ? ` · ${tags}` : ""}<span class="e-views" data-views="${entryPath(e)}" hidden></span></span>
    </a>
  </li>`;
}

function seriesCard(s, idx) {
  const eps = s.entries.map((slug) => idx.entries.find((e) => e.slug === slug));
  const first = eps[0], last = eps[eps.length - 1];
  const words = eps.reduce((n, e) => n + e.words, 0);
  return `<article class="series-card">
    <p class="sc-kicker">Series · ${eps.length} parts · ${Math.round(words / 220)} min</p>
    <h3><a href="journal.html?series=${encodeURIComponent(s.slug)}">${esc(s.title)}</a></h3>
    ${s.description ? `<p class="sc-desc">${esc(s.description)}</p>` : ""}
    <ol class="sc-steps">
      ${eps.map((e, i) => `<li><a href="${entryUrl(e)}"><span class="sc-n">${pad(i + 1, 2)}</span><span class="sc-t">${esc(shortTitle(e.title))}</span></a></li>`).join("")}
    </ol>
    <p class="sc-foot"><span>${fmtDate(first.date)} — ${fmtDate(last.date)}</span><a href="${entryUrl(first)}">Start reading →</a></p>
  </article>`;
}

async function journal() {
  const idx = await loadIndex();
  const params = new URLSearchParams(location.search);
  let active = params.get("tag");
  let activeSeries = params.get("series");
  let view = (() => { try { return localStorage.getItem("journal-view"); } catch { return null; } })() || "timeline";
  const allSeries = idx.series || [];

  if (allSeries.length) {
    $("#series").hidden = false;
    $("#series-grid").innerHTML = allSeries.map((s) => seriesCard(s, idx)).join("");
  }

  const tags = [...new Set(idx.entries.flatMap((e) => e.tags || []))].sort();
  const renderFilters = () => {
    const chip = (key, val, label, on) =>
      `<button class="tag${key === "series" ? " tag-series" : ""}" data-${key}="${esc(val)}" aria-pressed="${on}">${label}</button>`;
    $("#filters").innerHTML = [
      chip("tag", "all", "all", !active && !activeSeries),
      ...allSeries.map((s) => chip("series", s.slug, `∑ ${esc(s.title)}`, activeSeries === s.slug)),
      ...tags.map((t) => chip("tag", t, esc(t), t === active)),
    ].join("");
    document.querySelectorAll(".view-toggle [data-view]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  };

  const renderList = () => {
    const list = idx.entries.filter(
      (e) => (!active || (e.tags || []).includes(active)) && (!activeSeries || e.series?.slug === activeSeries)
    );
    if (!list.length) {
      $("#list").innerHTML = `<p class="empty">Nothing here yet.</p>`;
      return;
    }
    if (view === "timeline") {
      let month = "";
      const items = list.map((e) => {
        const m = monthLabel(e.date);
        const head = m !== month ? `<li class="tl-month"><span>${m}</span></li>` : "";
        month = m;
        return head + timelineItem(e);
      });
      $("#list").innerHTML = `<ol class="timeline">${items.join("")}<li class="tl-end" aria-hidden="true"><span>∎</span></li></ol>`;
    } else {
      const byYear = {};
      list.forEach((e) => (byYear[e.date.slice(0, 4)] ??= []).push(e));
      $("#list").innerHTML = Object.keys(byYear)
        .sort((a, b) => b - a)
        .map((y) => `<h2 class="year">${y}</h2><ol class="entries">${byYear[y].map((e) => entryRow(e, true)).join("")}</ol>`)
        .join("");
    }
    fillViews($("#list"));
  };

  const syncUrl = () => {
    const q = new URLSearchParams();
    if (active) q.set("tag", active);
    if (activeSeries) q.set("series", activeSeries);
    history.replaceState(null, "", q.toString() ? `?${q}` : location.pathname);
  };

  $("#filters").addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-tag],[data-series]");
    if (!btn) return;
    if (btn.dataset.series) {
      activeSeries = activeSeries === btn.dataset.series ? null : btn.dataset.series;
      active = null;
    } else {
      active = btn.dataset.tag === "all" ? null : btn.dataset.tag;
      if (btn.dataset.tag === "all") activeSeries = null;
    }
    syncUrl();
    renderFilters();
    renderList();
  });

  document.querySelector(".view-toggle").addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-view]");
    if (!btn) return;
    view = btn.dataset.view;
    try { localStorage.setItem("journal-view", view); } catch {}
    renderFilters();
    renderList();
  });

  $("#count").textContent = `${idx.entries.length} ${idx.entries.length === 1 ? "entry" : "entries"}`;
  renderFilters();
  renderList();
}

/* ───────── Single entry ───────── */
async function entry() {
  const slug = new URLSearchParams(location.search).get("e");
  const idx = await loadIndex();
  const i = idx.entries.findIndex((e) => e.slug === slug);
  if (i < 0) {
    $("#e-title").textContent = "Entry not found";
    $("#e-body").innerHTML = `<p><a href="journal.html">← Back to the journal</a></p>`;
    return;
  }
  const e = idx.entries[i];
  const res = await fetch(`journal/${encodeURIComponent(e.file)}`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${e.file} → ${res.status}`);
  const body = stripFrontMatter(await res.text());

  document.title = `${e.title} — Research Journal`;
  $("#e-no").textContent = `Entry No. ${pad(e.no)}`;
  $("#e-title").textContent = e.title;
  $("#e-meta").innerHTML = [
    `<time datetime="${e.date}">${fmtDate(e.date)}</time>`,
    `<span>${readTime(e)}</span>`,
    `<span data-views="${entryPath(e)}" hidden></span>`,
    ...(e.tags || []).map((t) => `<a href="journal.html?tag=${encodeURIComponent(t)}">#${esc(t)}</a>`),
    `<a href="#discussion">discuss ↓</a>`,
    `<a href="journal/${encodeURIComponent(e.file)}">source</a>`,
  ].join("");

  $("#e-body").innerHTML =
    e.type === "txt"
      ? `<pre class="plain">${esc(body.trim())}</pre>`
      : renderMarkdown(body.replace(/^\s*#\s+.*(\r?\n|$)/, "")); // title is already in the header

  // No drop cap when a post opens with a bold label like "TL;DR:"
  const lead = $("#e-body > p");
  if (lead?.firstChild?.nodeName === "STRONG") lead.classList.add("no-drop");

  let older = idx.entries[i + 1];
  let newer = idx.entries[i - 1];
  let [olderLabel, newerLabel] = ["← Older", "Newer →"];

  const series = e.series && (idx.series || []).find((s) => s.slug === e.series.slug);
  if (series) {
    const eps = series.entries.map((slug) => idx.entries.find((x) => x.slug === slug));
    const at = eps.findIndex((x) => x.slug === e.slug);
    $("#series-box").hidden = false;
    $("#series-box").innerHTML = `
      <p class="sb-head"><span>∑ Series</span><a href="journal.html?series=${encodeURIComponent(series.slug)}">${esc(series.title)}</a><span>Part ${at + 1} of ${eps.length}</span></p>
      <ol class="sb-steps" style="--n:${eps.length}">
        ${eps.map((x, k) => `<li class="${k < at ? "done" : k === at ? "current" : ""}"><a href="${entryUrl(x)}"${k === at ? ' aria-current="page"' : ""}><span class="sb-dot"></span><span class="sb-n">Part ${k + 1}</span><span class="sb-t">${esc(shortTitle(x.title))}</span></a></li>`).join("")}
      </ol>`;
    [older, newer] = [eps[at - 1], eps[at + 1]];
    [olderLabel, newerLabel] = ["← Previous in series", "Next in series →"];
  }
  $("#pager").innerHTML =
    (older ? `<a class="prev" href="${entryUrl(older)}"><small>${olderLabel}</small><span>${esc(older.title)}</span></a>` : "") +
    (newer ? `<a class="next" href="${entryUrl(newer)}"><small>${newerLabel}</small><span>${esc(newer.title)}</span></a>` : "");

  discussion(e).catch(() => {});
  trackView(entryPath(e), e.title);
  fillViews($("#e-meta"));

  const bar = $("#progress");
  const onScroll = () => {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    bar.style.width = `${max > 0 ? (h.scrollTop / max) * 100 : 0}%`;
  };
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

/* ───────── Comments, replies & upvotes (giscus → GitHub Discussions) ───────── */
const giscusTheme = () => (document.documentElement.dataset.theme === "dark" ? "noborder_dark" : "noborder_light");

function syncGiscusTheme() {
  const frame = document.querySelector("iframe.giscus-frame");
  frame?.contentWindow.postMessage({ giscus: { setConfig: { theme: giscusTheme() } } }, "https://giscus.app");
}

async function discussion(e) {
  const section = $("#discussion");
  const cfg = (await loadProfile()).comments || {};
  section.hidden = false;

  const toggle = $("#disc-toggle");
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    section.classList.toggle("collapsed", !open);
    $("#disc-toggle-text").textContent = open ? "Collapse all" : "Show discussion";
  });

  if (!cfg.repo || !cfg.repoId || !cfg.categoryId) {
    $(".giscus").innerHTML = `<p class="empty">Comments aren’t connected yet — add your giscus IDs to <code>content/profile.json</code> (see README).</p>`;
    return;
  }

  const s = document.createElement("script");
  Object.entries({
    src: "https://giscus.app/client.js",
    "data-repo": cfg.repo,
    "data-repo-id": cfg.repoId,
    "data-category": cfg.category,
    "data-category-id": cfg.categoryId,
    "data-mapping": "specific",
    "data-term": e.slug, // one discussion thread per entry, stable across domains
    "data-strict": "1",
    "data-reactions-enabled": "1",
    "data-emit-metadata": "0",
    "data-input-position": "top",
    "data-theme": giscusTheme(),
    "data-lang": "en",
    "data-loading": "lazy",
    crossorigin: "anonymous",
  }).forEach(([k, v]) => s.setAttribute(k, v));
  s.async = true;
  $(".giscus").after(s);
}

/* ───────── Ambient math background ───────── */
function mathBackground() {
  const glyphs = ["∑", "∫", "∂", "∇", "∞", "π", "λ", "θ", "∮", "√", "≈", "∈", "⊗", "ℏ", "Δ", "Ω", "φ", "μ", "σ", "ε", "∀", "∃", "ℝ", "ψ", "ζ", "∝", "ξ"];
  const equations = [
    "e<sup>iπ</sup> + 1 = 0",
    "∫ e<sup>−x²</sup> dx = √π",
    "P(A|B) = P(B|A) P(A) / P(B)",
    "∇ · E = ρ / ε₀",
    "det(A − λI) = 0",
    "H(X) = −Σ p log p",
    "f(x) = Σ aₙ xⁿ",
    "iℏ ∂ψ/∂t = Ĥψ",
    "x̂ = argmin ‖Ax − b‖²",
    "∂L/∂θ = 0",
  ];

  let seed = 20261013;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const pick = (a) => a[Math.floor(rnd() * a.length)];
  const html = [];
  const span = (cls, x, y, size, opacity, content) => {
    const vars = [
      `left:${x.toFixed(1)}%`, `top:${y.toFixed(1)}%`, `--s:${size}px`, `--o:${opacity.toFixed(3)}`,
      `--r:${(rnd() * 16 - 8).toFixed(1)}deg`, `--dx:${(rnd() * 20 - 10).toFixed(0)}px`,
      `--d:${(16 + rnd() * 18).toFixed(1)}s`, `--delay:${(-rnd() * 20).toFixed(1)}s`,
    ];
    html.push(`<span class="${cls}" style="${vars.join(";")}">${content}</span>`);
  };

  // Single glyphs on a jittered grid, so they spread evenly without clumping.
  const cols = 7, rows = 7;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      if (rnd() < 0.2) continue;
      const x = ((c + 0.15 + rnd() * 0.7) / cols) * 100;
      const big = (x < 13 || x > 90) && rnd() < 0.35; // large accent glyphs only in the outer margins
      span(big ? "hl" : "", x, ((r + 0.15 + rnd() * 0.7) / rows) * 100,
        big ? 64 + rnd() * 40 : 18 + rnd() * 26, big ? 0.12 : 0.11 + rnd() * 0.09, pick(glyphs));
    }

  // Whole equations hug the left and right margins.
  equations.forEach((eq, i) => {
    const left = i % 2 === 0;
    span("eq", left ? 1.5 + rnd() * 9 : 74 + rnd() * 9, 9 + (i / equations.length) * 86 + rnd() * 3,
      15 + rnd() * 6, 0.2 + rnd() * 0.08, eq);
  });

  const field = document.createElement("div");
  field.className = "mathfield";
  field.setAttribute("aria-hidden", "true");
  field.innerHTML = html.join("");
  document.body.prepend(field);

  // Gentle parallax: the symbols drift slower than the page.
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let ticking = false;
  addEventListener("scroll", () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const max = innerHeight * 0.35;
      field.style.setProperty("--par", `${-Math.min(scrollY * 0.05, max)}px`);
      ticking = false;
    });
  }, { passive: true });
}

/* ───────── Theme toggle ───────── */
function themeToggle() {
  const root = document.documentElement;
  const btn = $("#theme-toggle");
  const set = (t, animate) => {
    if (animate) {
      root.classList.add("theme-anim");
      setTimeout(() => root.classList.remove("theme-anim"), 600);
    }
    root.dataset.theme = t;
    syncGiscusTheme();
  };
  btn?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("theme", next); } catch {}
    set(next, true);
  });
  // Follow the system setting until the visitor picks a theme explicitly.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    let saved = null;
    try { saved = localStorage.getItem("theme"); } catch {}
    if (!saved) set(e.matches ? "dark" : "light", true);
  });
}

/* ───────── Boot ───────── */
const pages = { home, journal, entry };
document.addEventListener("DOMContentLoaded", () => {
  mathBackground();
  themeToggle();
  $("#year") && ($("#year").textContent = new Date().getFullYear());
  const run = pages[document.body.dataset.page];
  run?.().catch((err) => showError($("main"), err));
  const page = document.body.dataset.page;
  if (page === "home") trackView("/", "Home");
  if (page === "journal") trackView("/journal", "Journal index");
});
