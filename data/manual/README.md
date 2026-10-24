Manual downloads
================

Files placed here are ingested by `python -m quantrag.ingest` when they are
listed in `../manual_sources.csv`:

    title,url,category,file,priority_tier

- `file` is resolved relative to this directory.
- The document is cited by `url`, not by the local file path.
- Use this for sources where robots.txt or an interactive download blocks the
  crawler (for example `https://www.columbia.edu/~ks20/6712-14/6712-14-Notes-B-Scholes.pdf`).

Nothing in this directory is committed except this note (see `.gitignore`).
