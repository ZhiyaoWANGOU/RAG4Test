#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
page_to_sections.py

功能：
1) 请求网页 HTML
2) 用 Readability 抽取正文 HTML
3) 转 Markdown（保留标题/列表/链接）
4) 按 Markdown 标题(#..######)切分为结构化 JSON

用法：
  python page_to_sections.py --url "https://support.mozilla.org/en-US/kb/back-and-restore-information-firefox-profiles" \
                             --levels 1 2
说明：
  --levels 可选，指定要保留的标题层级（例如 1 2 只保留 # 和 ##）。若不传，保留所有层级。
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from typing import List, Dict

import requests
from readabilipy import simple_json_from_html_string
from markdownify import markdownify as md

HEADERS = {"User-Agent": "RAG4Test/0.1 (+readability-to-sections)"}


def log(msg: str) -> None:
    print(f"[rtm] {msg}", file=sys.stderr)


def fetch_html(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout, proxies={"http": None, "https": None})
    r.raise_for_status()
    return r.text


def html_to_markdown(html: str) -> str:
    # 保留标题/列表/链接；压缩多余空行
    md_text = md(html, strip=["script", "style"], heading_style="ATX")
    md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
    return md_text


def split_markdown(md_text: str, keep_levels: List[int] | None = None) -> List[Dict]:
    """
    将 Markdown 按标题切分为 {title, level, text} 段。
    keep_levels: 仅保留指定层级（如 [1,2]）；None 则保留全部。
    """
    heading_re = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)

    sections: List[Dict] = []
    last_idx = 0
    last_heading = None  # (level, title)

    for m in heading_re.finditer(md_text):
        hashes, title = m.groups()
        level = len(hashes)
        start = m.end()

        if last_heading is not None:
            body = md_text[last_idx:m.start()].strip()
            if keep_levels is None or last_heading[0] in keep_levels:
                sections.append({
                    "title": last_heading[1],
                    "level": last_heading[0],
                    "text": body
                })

        last_heading = (level, title.strip())
        last_idx = start

    if last_heading is not None:
        body = md_text[last_idx:].strip()
        if keep_levels is None or last_heading[0] in keep_levels:
            sections.append({
                "title": last_heading[1],
                "level": last_heading[0],
                "text": body
            })

    return sections


def main():
    ap = argparse.ArgumentParser(description="Fetch page -> Readability -> Markdown -> JSON sections")
    ap.add_argument("--url", required=True, help="Page URL")
    ap.add_argument("--levels", nargs="*", type=int, default=None,
                    help="Heading levels to keep (e.g., 1 2). Default: keep all")
    ap.add_argument("--print-md", action="store_true", help="Also print the extracted Markdown to stderr for debugging")
    args = ap.parse_args()

    log(f"fetching: {args.url}")
    html = fetch_html(args.url)
    log(f"html length: {len(html)}")

    log("running Readability…")
    data = simple_json_from_html_string(html, use_readability=True)
    content_html = (data.get("content") or "").strip()
    title = (data.get("title") or "").strip() or args.url

    if not content_html:
        log("no content extracted (Readability returned empty)")
        print(json.dumps({"doc": {"url": args.url, "title": title}, "sections": []}, ensure_ascii=False))
        return

    md_text = html_to_markdown(content_html)
    if args.print_md:
        log("--- MARKDOWN (preview) ---")
        for line in md_text.splitlines()[:80]:
            log(line)
        log("--- END MARKDOWN PREVIEW ---")

    sections = split_markdown(md_text, keep_levels=args.levels)

    out = {
        "doc": {"url": args.url, "title": title},
        "sections": sections
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
