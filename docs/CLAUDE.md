# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Jekyll site for the **IMPSY** (Intelligent Musical Prediction System) project. It lives in the `docs/` directory of the main `cpmpercussion/impsy` repository and deploys to GitHub Pages at `https://charlesmartin.au/impsy`. The site is a small marketing-free landing zone for researchers and music technologists who want to adopt or adapt IMPSY — see `README.md` for the editorial brief.

This is a self-contained Jekyll project: run all of the commands below from inside `docs/`, not the repository root.

## Commands

```bash
cd docs
bundle install                  # install Ruby gems (first run / Gemfile changes)
bundle exec jekyll serve        # local preview at http://localhost:4000/impsy/
bundle exec jekyll build        # one-shot build into docs/_site/
```

`_config.yml` is **not** reloaded by `jekyll serve` — restart the server after editing it.

## Deploy

Pushes to `main` trigger `.github/workflows/pages.yml` (at the repo root), which builds this Jekyll site from `docs/` with the version pinned in `Gemfile` and publishes via `actions/deploy-pages`. The build runs in `working-directory: docs` and uploads `docs/_site`. The workflow injects `--baseurl` from `actions/configure-pages`, so the static `baseurl: "/impsy"` in `_config.yml` is for local builds. The deploy is independent of the Python test workflow (`python-app.yml`) — the site publishes even if tests are red.

For Pages to pick up the workflow, the repo's *Settings → Pages → Source* must be set to **GitHub Actions** (not "Deploy from a branch").

## Architecture

- **Pages are `.md` files in `docs/`** with explicit `permalink:` (`index.md`, `get-started.md`, `config.md`, `research.md`, `gallery.md`, `404.html`). There is no `_posts/` directory and no blog — this is intentional per the brief; do not reinstate it. `config.md` is the IMPSY configuration reference, kept in sync with the real `config.toml` schema in the parent repo.
- **Layouts** live in `_layouts/`: `default.html` is the shell (Bootstrap 5 CDN, SEO tag, theme bootstrapper); `home.html` is a thin wrapper used by `index.md` so the hero can use bespoke markup; `page.html` is the standard article wrapper used by all content pages.
- **Includes** in `_includes/`: `header.html` (nav driven by `site.nav` in `_config.yml`, plus the theme toggle button) and `footer.html` (driven by `site.links`).
- **Styling**: `assets/css/main.scss` (front-matter triggers Jekyll SCSS compilation to `main.css`). Bootstrap loads via CDN; the SCSS adds an accent palette, hero, workflow cards, timeline, publication list, and gallery grid. Component CSS classes used in markdown — `hero`, `workflow-step`, `timeline`, `pub-list`, `gallery-grid`, `video-grid`, `btn-impsy` — are defined here.
- **Theme switching**: `assets/js/theme-toggle.js` cycles auto → light → dark, persists to `localStorage`, and writes `data-bs-theme` on `<html>`. An inline script in `_layouts/default.html` applies the stored choice before paint to prevent FOUC. Both scripts must agree on the storage key (`impsy-theme`).
- **SEO/discovery**: `jekyll-seo-tag` reads each page's `title`, `description`, and `image` front matter to emit Open Graph, Twitter, and JSON-LD. Always set these on new pages. `jekyll-sitemap` generates `sitemap.xml` using `site.url` + `site.baseurl`.

## Editorial conventions (from README)

- Voice is research-oriented NIME-community, not marketing. Link to source code and papers rather than describing them in promotional terms.
- Images come from the paper repositories under `cpmpercussion/*` (mainly `impsypi-opening-design-space-paper/figures/`). Don't introduce stock photos or generated graphics.
- Keep the site to a small set of clearly-written pages. Long-form news belongs on `charlesmartin.au` or `smcclab.au`, both linked in `_config.yml` under `site.links`.

## Adding a new page

1. Create `<slug>.md` in `docs/` with `layout: page`, `title:`, `subtitle:` (optional), `permalink: /<slug>/`, and `description:` (used by SEO).
2. Add an entry to `nav:` in `_config.yml` to surface it in the header.
3. Reuse existing component classes from `main.scss` before adding new ones.
