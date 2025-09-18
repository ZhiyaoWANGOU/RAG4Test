#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_search.py
Search Stack Overflow & Reddit for troubleshooting evidence.

Usage:
  python web_search.py --q "firefox search slow after update 131" --topk 5
  python web_search.py --q "restore profile to a different location" --topk 4 --out hits.json

Deps (choose one):
  pip install ddgs
  # or for older environments:
  pip install "duckduckgo-search>=5,<9"
"""

from __future__ import annotations
import argparse, json, re, sys
from urllib.parse import urlparse, urlunparse

# ---- ddg client import (兼容新旧包名) ----
DDGS = None
err_hint = ""
try:
    from ddgs import DDGS  # new name
except Exception:
    try:
        from duckduckgo_search import DDGS  # old name
        err_hint = "RuntimeWarning from duckduckgo_search is OK; you can switch to `pip install ddgs` later."
    except Exception as e:
        print("[fatal] neither `ddgs` nor `duckduckgo_search` is installed.", file=sys.stderr)
        print("Try: pip install ddgs", file=sys.stderr)
        sys.exit(2)

STACK = "stackoverflow.com"
REDDIT = "reddit.com"

def normalize_url(u: str) -> str:
    """Drop tracking/query/fragments; keep scheme+netloc+path for dedupe."""
    try:
        p = urlparse(u)
        # convert old reddit mobile etc. to canonical host
        host = p.netloc.lower()
        host = host.replace("old.reddit.com", "www.reddit.com")
        host = host.replace("np.reddit.com", "www.reddit.com")
        host = host.replace("m.reddit.com", "www.reddit.com")
        host = host.replace("r.", "www.") if host.endswith("reddit.com") and host.startswith("r.") else host
        host = host.replace("stackprinter.appspot.com", STACK)
        path = re.sub(r"/+$", "", p.path)
        return urlunparse((p.scheme or "https", host, path, "", "", ""))
    except Exception:
        return u

def dedupe(items):
    seen = set()
    out = []
    for it in items:
        key = (normalize_url(it["url"]), (it.get("title") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def search_once(ddg: DDGS, query: str, site: str, max_results: int):
    q = f"{query} site:{site}"
    results = []
    for r in ddg.text(q, max_results=max_results):
        url = r.get("href") or r.get("url") or ""
        title = r.get("title") or ""
        snippet = r.get("body") or r.get("snippet") or ""
        if not url or not title:
            continue
        results.append({
            "source": "stackoverflow" if STACK in url else ("reddit" if "reddit.com" in url else site),
            "title": title.strip(),
            "url": url.strip(),
            "snippet": snippet.strip()
        })
    return results

def run_search(query: str, topk_per_site: int):
    if err_hint:
        print(f"[warn] {err_hint}", file=sys.stderr)
    items = []
    with DDGS() as ddg:
        items += search_once(ddg, query, STACK, topk_per_site)
        items += search_once(ddg, query, REDDIT, topk_per_site)
    return dedupe(items)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", "--query", dest="q", required=True, help="search query")
    ap.add_argument("--topk", type=int, default=5, help="per-site max results")
    ap.add_argument("--out", default="", help="optional JSON output file")
    args = ap.parse_args()

    hits = run_search(args.q, args.topk)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"query": args.q, "results": hits}, f, ensure_ascii=False, indent=2)
        print(f"✅ saved {len(hits)} results to {args.out}")
    else:
        print(json.dumps({"query": args.q, "results": hits}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
