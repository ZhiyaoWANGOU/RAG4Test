#!/usr/bin/env python3
"""
Knowledge Base Builder for Firefox Support

Usage:
  Discover article URLs:
    python kb_builder.py --mode discover \
      --seeds https://support.mozilla.org/en-US/kb/ \
      --out urls.txt --max-pages 500 --delay 1.0

  Build structured JSON from URLs:
    python kb_builder.py --mode build \
      --urls urls.txt --outdir ./kb_json --levels 1 2 --delay 1.5
"""

import argparse
import re
import sys
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path
from readability import Document

# ---------------- Config ---------------- #

KB_RE = re.compile(r"^https://support\.mozilla\.org/en-US/kb/[^/?#]+$")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0 Safari/537.36"
}

# ---------------- Discover Mode ---------------- #

def discover(seed_urls, out_path, max_pages=500, delay=1.0):
    seen = set()
    q = list(seed_urls)
    out = open(out_path, "w", encoding="utf-8")

    count = 0
    while q and count < max_pages:
        url = q.pop(0)
        if url in seen:
            continue
        seen.add(url)
        count += 1
        print(f"[kb] [{count}/{max_pages}] GET {url}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[err] {url}: {e}")
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"])
            # 只保留 /en-US/kb/ 的文章
            if KB_RE.match(href) and href not in seen:
                q.append(href)
                out.write(href + "\n")
        time.sleep(delay)

    out.close()
    print(f"[done] wrote {out_path} with {count} pages")

# ---------------- Build Mode ---------------- #

def extract_sections(url, levels=(1,2)):
    print(f"[kb] parsing: {url}")
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"[err] {url}: {e}")
        return None

    doc = Document(r.text)
    title = doc.short_title()
    soup = BeautifulSoup(doc.summary(), "html.parser")

    results = []
    for tag in soup.find_all([f"h{lv}" for lv in levels]):
        level = int(tag.name[1])
        title_text = tag.get_text(strip=True)
        texts = []
        for sib in tag.find_next_siblings():
            if sib.name and sib.name.startswith("h") and int(sib.name[1]) <= level:
                break
            texts.append(sib.get_text(" ", strip=True))
        body = "\n".join(t for t in texts if t)
        if body.strip():
            results.append({
                "title": title_text,
                "level": level,
                "text": body
            })

    return {
        "url": url,
        "title": title,
        "sections": results
    }

def build(urls_file, outdir, levels=(1,2), delay=1.5):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in open(urls_file, encoding="utf-8") if u.strip()]
    for i, url in enumerate(urls, 1):
        data = extract_sections(url, levels=levels)
        if not data:
            continue
        out_path = Path(outdir) / f"doc_{i:04d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[ok] {out_path}")
        time.sleep(delay)

# ---------------- CLI ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["discover", "build"], required=True)
    ap.add_argument("--seeds", nargs="+", help="Seed URLs (for discover)")
    ap.add_argument("--out", help="Output file (for discover)")
    ap.add_argument("--urls", help="URLs file (for build)")
    ap.add_argument("--outdir", help="Output dir (for build)")
    ap.add_argument("--levels", nargs="+", type=int, default=[1,2])
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    if args.mode == "discover":
        if not args.seeds or not args.out:
            print("discover mode requires --seeds and --out")
            sys.exit(1)
        discover(args.seeds, args.out, args.max_pages, args.delay)

    elif args.mode == "build":
        if not args.urls or not args.outdir:
            print("build mode requires --urls and --outdir")
            sys.exit(1)
        build(args.urls, args.outdir, tuple(args.levels), args.delay)

if __name__ == "__main__":
    main()
