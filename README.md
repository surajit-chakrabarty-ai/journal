# Research Journal — Surajit Chakrabarty

A minimal, static research journal hosted free on GitHub Pages. It has no framework and no build tools to install: plain HTML, CSS and JavaScript, plus two small Node scripts.

**Live site:** https://surajit-chakrabarty-ai.github.io (later `https://chakrsur.ai`, see [Custom domain](#custom-domain-chakrsurai))

### Features
- **Landing page:** portrait, bio, links, current projects, recent entries, and a side panel of quick facts.
- **Journal:** a **Timeline / List** switch, tag and series filters, and **series** cards (e.g. *Inference Engineering*, Parts 1–4).
- **Reader:** Markdown with KaTeX math (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`), tables, code, a reading-progress bar, a series stepper, and previous/next links.
- **Comments, replies and upvotes** through giscus (GitHub Discussions), with a *Collapse all* button.
- **View counts** through GoatCounter (cookie-less).
- **Light/dark mode** that follows the system setting, with a manual toggle that is remembered.
- Ambient math-symbol background and a lab-notebook look (Instrument Serif, Newsreader, IBM Plex Mono).

---

## Project structure

```
index.html                 Landing page
journal.html               Journal index (timeline / list, series, filters)
entry.html                 Single-entry reader + discussion
assets/style.css           All styling (light + dark themes)
assets/app.js              All client-side logic
assets/me.jpg              Your portrait (assets/portrait.svg = illustrated fallback)
content/profile.json       ← name, bio, links, projects, photo, comments + analytics config
content/journal.json       ← per-post dates/tags/titles + series definitions
journal/                   Published posts (.md / .txt) + generated index.json
scripts/sync-blogs.mjs     Copies posts from ~/Documents/Blogs into journal/ and rebuilds the index
scripts/build-index.mjs    Builds journal/index.json (also runs on every push)
.github/workflows/deploy.yml   GitHub Actions: build index → deploy to Pages
CNAME                      Custom domain (only used once it's set in GitHub Settings → Pages)
```

---

## Preview locally

```bash
cd ~/Documents/ResearchJournal
node scripts/build-index.mjs && python3 -m http.server 8000
```
Open http://localhost:8000 and press `Ctrl+C` to stop.

- **"Address already in use":** run `lsof -ti:8000 | xargs kill`, or use another port (`python3 -m http.server 8001`).
- **Changes not showing:** hard-refresh with `Cmd+Shift+R`.
- Opening the HTML files directly (`file://`) won't work, because the browser blocks loading the content files.

---

## Hosting on GitHub (account: `Surajit-Chakrabarty-ai`)

GitHub Free hosts Pages only from **public** repos.

**Before the first push**
- Set the real publish dates in `content/journal.json`. The current ones are placeholders.
- Delete `journal/_archive/` if you don't need the old sample posts.
- Run `node scripts/sync-blogs.mjs` to pull in your latest posts.

**1. Create the repo.** Go to https://github.com/new and enter:
- Name: **`surajit-chakrabarty-ai.github.io`**. It must be exactly this, in lowercase.
- Visibility: **Public**.
- Leave "Add README / .gitignore / license" unticked.

**2. Push the project**
```bash
cd ~/Documents/ResearchJournal
git init -b main
git add .
git commit -m "Research journal"
git remote add origin https://github.com/Surajit-Chakrabarty-ai/surajit-chakrabarty-ai.github.io.git
git push -u origin main
```
When git asks for a password, use a **personal access token**, not your GitHub password. Create one at GitHub → Settings → Developer settings → Personal access tokens → *Tokens (classic)*, with the `repo` and `workflow` scopes.
Or, easier: run `brew install gh && gh auth login` once, and `git push` will just work.

**3. Turn on Pages.** In the repo, go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.

**4. Deploy.** Open the **Actions** tab. The first "Publish journal" run usually fails because Pages wasn't enabled yet, so open it and click **Re-run all jobs**. When it shows a green tick (about a minute), the site is live at https://surajit-chakrabarty-ai.github.io.

Every later push to `main` redeploys automatically.

---

## Writing and publishing

### From `~/Documents/Blogs` (recommended)
Write in `~/Documents/Blogs`, then run:
```bash
node scripts/sync-blogs.mjs
git add . && git commit -m "New post" && git push
```
- Copies `.md` / `.txt` posts and their images into `journal/`. Your originals are never modified.
- Stamps each copy with a `date:` from the file's last-modified time, unless the post already sets one. A date in `content/journal.json` overrides both.
- Posts deleted from Blogs are removed from the journal on the next sync.
- To sync from a different folder, run `node scripts/sync-blogs.mjs /path/to/folder`.

### Directly in `journal/`
Add `journal/YYYY-MM-DD-title.md` (or `.txt`) and push. Front matter is optional:
```
---
title: My result
date: 2026-10-12
tags: [evaluation, LLMs]
summary: One line shown in the index.
series: Inference Engineering
episode: 5
draft: false
---
```
Without front matter, the title comes from the first `# heading` (or the first line of a `.txt` file), the date from the filename, and the summary from the first paragraph. A leading "TL;DR:" is stripped from the summary.

- **Drafts:** `draft: true`, or a filename starting with `_`, keeps the post off the site. The repo is public, though, so drafts can still be read on GitHub.
- **Math:** `$inline$`, `$$display$$`, `\(inline\)`, `\[display\]`.
- **Images:** keep them next to the post (e.g. `img/fig.png`) and reference them relatively as `![caption](img/fig.png)`.
- **Plain `.txt`** files are shown exactly as written, in a monospace font.

---

## Configuration

### `content/profile.json`: you and your site
| Key | What it does |
|-----|--------------|
| `name`, `role`, `bio[]` | Landing page heading and bio. The bio supports inline Markdown. The last word of the name is set in italics on its own line. |
| `photo` | Portrait path, e.g. `assets/me.jpg`, ideally 4:5 and at least 360×450. |
| `links[]` | `{ "label", "url" }` links under the bio. |
| `notes[]` | `["Label", "Value"]` pairs in the side panel. Entry count, last entry and total views are added automatically. |
| `projects[]` | `{ title, status: active\|paused\|done, summary, tags[], since, link }` |
| `comments` | giscus settings, see below. |
| `analytics` | GoatCounter settings, see below. |

### `content/journal.json`: dates, tags and series
Overrides any post's `date`, `tags`, `title` or `summary` without touching the original, and defines **series**:
```json
{
  "series": [
    { "slug": "inference-engineering", "title": "Inference Engineering",
      "description": "A four-part deep dive into serving LLMs…",
      "entries": ["llm-inference-ep1-the-token-factory", "llm-inference-ep2-the-batch-whisperer",
                  "llm-inference-ep3-the-distributed-orchestra", "llm-inference-ep4-the-capacity-planner"] }
  ],
  "entries": {
    "llm-inference-ep1-the-token-factory": { "date": "2026-09-22", "tags": ["inference", "systems"] }
  }
}
```
Keys are file names without the extension. Series entries are listed in reading order. Each series gets a card on the journal page, a ∑ filter chip, a "Part N of M" stepper on every episode, and "Previous/Next in series" links.

---

## Comments, replies and upvotes (giscus, free)

Comments live in the repo's **GitHub Discussions**. Readers sign in with GitHub to comment, reply in threads, or react. A 👍 reaction on the post or on a comment works as an upvote.

1. Repo → **Settings → General → Features** → tick **Discussions**.
2. Install the giscus app on this repo only: https://github.com/apps/giscus
3. At https://giscus.app, enter `Surajit-Chakrabarty-ai/surajit-chakrabarty-ai.github.io` and choose the category **Announcements**.
4. Copy `data-repo-id` and `data-category-id` into `content/profile.json`:
   ```json
   "comments": {
     "repo": "Surajit-Chakrabarty-ai/surajit-chakrabarty-ai.github.io",
     "repoId": "R_kgDO…", "category": "Announcements", "categoryId": "DIC_kwDO…"
   }
   ```
5. Push. Each post gets its own thread, keyed by file name, so renaming a file starts a new thread. You moderate from the repo's **Discussions** tab.

## Page views (GoatCounter, free)

1. Sign up at https://www.goatcounter.com/signup and pick a code, e.g. `chakrsur` (giving `chakrsur.goatcounter.com`).
2. In GoatCounter **Settings**, tick **"Allow adding visitor counts on your website"**.
3. In `content/profile.json`, set `"analytics": { "goatcounter": "chakrsur", "showViews": true }` and push.

Views appear on each entry, in the journal list, and as "Total views" on the landing page. Set `"showViews": false` to track views without displaying them. Local previews aren't counted, and some ad blockers block counting. The full dashboard is at `https://<code>.goatcounter.com`.

---

## Customising

- **Colours and fonts:** CSS variables at the top of `assets/style.css`, with the light theme in `:root` and the dark theme in `:root[data-theme="dark"]`.
- **Background symbols:** `mathBackground()` in `assets/app.js` holds the glyph and equation lists and their opacity (`0.11 + rnd() * 0.09` for single symbols, `0.2 + rnd() * 0.08` for equations). The `.mathfield` mask in `style.css` controls how much shows behind the text.
- **Cache busting:** after editing `style.css` or `app.js`, bump `?v=N` in the three HTML files so visitors get the new version.

---

## Custom domain (chakrsur.ai)

GitHub hosting is free, but the domain must be bought from a registrar (Cloudflare, Namecheap, Porkbun…). `.ai` domains cost about US$70–100 a year, often with a 2-year minimum.

1. **Verify the domain (prevents takeover):** GitHub → your profile **Settings → Pages → Add a domain** → `chakrsur.ai`. Add the `TXT` record it shows at your registrar, then click Verify.
2. **Add DNS records** at your registrar:

   | Type  | Name | Value |
   |-------|------|-------|
   | A     | @    | 185.199.108.153 |
   | A     | @    | 185.199.109.153 |
   | A     | @    | 185.199.110.153 |
   | A     | @    | 185.199.111.153 |
   | AAAA  | @    | 2606:50c0:8000::153 |
   | AAAA  | @    | 2606:50c0:8001::153 |
   | AAAA  | @    | 2606:50c0:8002::153 |
   | AAAA  | @    | 2606:50c0:8003::153 |
   | CNAME | www  | `surajit-chakrabarty-ai.github.io` |

   Remove any parking/forwarding records. On Cloudflare, use **DNS only** (grey cloud).
3. Repo → **Settings → Pages → Custom domain** → `chakrsur.ai` → Save. Once the DNS check passes (minutes to a few hours), tick **Enforce HTTPS**.
4. Check it:
   ```bash
   dig chakrsur.ai +short        # → 185.199.108–111.153
   dig www.chakrsur.ai +short    # → surajit-chakrabarty-ai.github.io
   ```

If you set up a custom domain later, add the new domain at giscus.app too, and in GoatCounter if you restrict allowed origins there.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| First Actions run failed | Settings → Pages → Source = **GitHub Actions**, then **Re-run all jobs**. |
| `git push` rejects the password | Use a personal access token, or `gh auth login`. |
| New post not showing | Run `node scripts/sync-blogs.mjs` (or `build-index.mjs`), then commit and push. |
| Old styles after an update | `Cmd+Shift+R`, and bump `?v=N` in the HTML files. |
| "Comments aren't connected yet" | Fill in `repoId` / `categoryId` in `content/profile.json`. |
| No view counts | Set the GoatCounter code and enable "Allow adding visitor counts". |
| Math shows raw `\(`…`\)` | Check every opening delimiter has a matching close (`\(`…`\)`, `\[`…`\]`, `$`…`$`). |
