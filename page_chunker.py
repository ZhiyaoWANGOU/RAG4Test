#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
page_chunker.py

- 仅抓取主内容容器：article.sumo-article .article-content（有回退）
- 跳过目录/页脚/语言切换等噪声区域
- 以 H2/H3 为分段边界；保留第一个 H2 之前的导语
- 沿文档流收集正文，直到下一个 H2/H3
- 按字符长度滑窗切块（带 overlap），输出 JSON

用法：
  python page_chunker.py --url "https://support.mozilla.org/en-US/kb/back-and-restore-information-firefox-profiles" \
                         --chunk 900 --overlap 150 --min-len 160
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

HEADERS = {"User-Agent": "RAG4Test/0.1 (+page-chunker)"}

# ----------------------------- logging helper ----------------------------- #
def log(msg: str) -> None:
    print(f"[chunker] {msg}", file=sys.stderr)

# ----------------------------- SUMO 噪声过滤 ------------------------------ #
NOISE_IDS = {
    "toc", "table-of-contents", "site-footer", "footer", "language",
    "languages"
}
NOISE_CLASSES = {
    "toc", "sumo-toc", "site-footer", "footer", "language-switcher",
    "mzp-c-footer", "c-footer", "l10n", "mzp-l-language-switcher"
}
NOISE_TITLES = {"table of contents", "volunteer"}

def _has_noise_ancestor(tag: Tag) -> bool:
    cur = tag
    while isinstance(cur, Tag):
        tid = (cur.get("id") or "").lower()
        if tid in NOISE_IDS:
            return True
        classes = [c.lower() for c in cur.get("class", [])]
        if any(c in NOISE_CLASSES for c in classes):
            return True
        cur = cur.parent
    return False

# ----------------------------- http & text utils -------------------------- #
def fetch_html(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout,
                     proxies={"http": None, "https": None})
    r.raise_for_status()
    return r.text

def _slugify(text: str) -> str:
    s = re.sub(r"\s+", "-", text.strip().lower())
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "section"

def _clean_text(text: str) -> str:
    text = re.sub(r"\r?\n|\t", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _extract_section_text(start: Tag, stop_tags: set[str]) -> str:
    """
    沿文档流（next_elements）收集 start 标题之后的文本，
    遇到下一个 H2/H3（且不是自身）即停止；跳过目录/页脚/语言切换等噪声区域。
    """
    chunks: list[str] = []
    for el in start.next_elements:
        if isinstance(el, Tag) and el.name in stop_tags and el is not start:
            break
        if isinstance(el, Tag):
            if el.name in {"nav", "aside", "script", "style", "noscript", "footer", "header"}:
                continue
            if _has_noise_ancestor(el):
                continue
            txt = el.get_text(separator=" ", strip=True)
        elif isinstance(el, NavigableString):
            parent = el.parent if isinstance(el.parent, Tag) else None
            if parent is not None and _has_noise_ancestor(parent):
                continue
            txt = str(el).strip()
        else:
            txt = ""
        txt = _clean_text(txt)
        if txt:
            chunks.append(txt)
    return _clean_text(" ".join(chunks))

def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = _clean_text(text)
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + size, n)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out

# ----------------------------- core parser -------------------------------- #
def parse_page(url: str, chunk: int, overlap: int, min_len: int) -> dict:
    log(f"fetching: {url}")
    html = fetch_html(url)
    if not html:
        log("html fetch failed (empty response)")
        return {
            "doc": {"url": url, "title": url, "section_count": 0, "chunk_count": 0},
            "chunks": []
        }

    log(f"html length: {len(html)}")
    soup = BeautifulSoup(html, "html.parser")

    # title
    title = soup.title.get_text(strip=True) if soup.title else url

    # 主内容容器（更窄，减少抓到目录/页脚的概率）
    main = soup.select_one("article.sumo-article .article-content") \
        or soup.select_one("article .article-content") \
        or soup.select_one("article.sumo-article") \
        or soup.select_one("article, .article, .sumo-article, main") \
        or soup.body or soup

    # 只在主容器内找 H2/H3
    headings = [h for h in main.find_all(["h2", "h3"]) if isinstance(h, Tag)]
    # 过滤位于噪声祖先下的 heading
    headings = [h for h in headings if not _has_noise_ancestor(h)]
    log(f"headings: {len(headings)}")
    stop_tags = {"h2", "h3"}

    sections: list[dict] = []

    # 导语：第一个 H2/H3 之前的正文
    if headings:
        first_h_text = headings[0].get_text(strip=True)
        full_text = _clean_text(main.get_text(separator=" ", strip=True))
        idx = full_text.find(first_h_text)
        if idx > 0:
            lead_text = _clean_text(full_text[:idx])
            if lead_text and len(lead_text) >= min_len:
                log(f"section 'Introduction' len={len(lead_text)}")
                sections.append({"anchor": "", "section_title": "Introduction", "text": lead_text})

    if not headings:
        body_text = _clean_text(main.get_text(separator=" ", strip=True))
        if body_text and len(body_text) >= min_len:
            log(f"section '{title}' len={len(body_text)}")
            sections.append({"anchor": "", "section_title": title, "text": body_text})
    else:
        for h in headings:
            sec_title = h.get_text(separator=" ", strip=True)
            low = sec_title.strip().lower()
            # 跳过目录/志愿者等非正文小节
            if low in NOISE_TITLES:
                log(f"skip heading (noise): {sec_title}")
                continue
            if _has_noise_ancestor(h):
                log(f"skip heading (noise ancestor): {sec_title}")
                continue

            anchor_id = h.get("id") or _slugify(sec_title)
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            anchor_url = f"{base}#{anchor_id}"

            sec_text = _extract_section_text(h, stop_tags)
            log(f"section '{sec_title}' len={len(sec_text)}")
            if len(sec_text) >= min_len:
                sections.append({
                    "anchor": anchor_url,
                    "section_title": sec_title,
                    "text": sec_text
                })

    log(f"sections kept (>= {min_len} chars): {len(sections)}")

    # 切块
    chunks: list[dict] = []
    for sec_idx, sec in enumerate(sections):
        parts = _chunk_text(sec["text"], size=chunk, overlap=overlap)
        for i, p in enumerate(parts):
            chunks.append({
                "section_index": sec_idx,
                "chunk_index": i,
                "anchor": sec["anchor"],
                "section_title": sec["section_title"],
                "text": p
            })

    log(f"chunks: {len(chunks)}")

    return {
        "doc": {
            "url": url,
            "title": title,
            "section_count": len(sections),
            "chunk_count": len(chunks)
        },
        "chunks": chunks
    }

# ----------------------------- CLI ---------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Split a support page into sectioned chunks with anchors.")
    ap.add_argument("--url", required=True)
    ap.add_argument("--chunk", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--min-len", type=int, default=160, help="min section text length to keep")
    args = ap.parse_args()

    data = parse_page(args.url, args.chunk, args.overlap, args.min_len)
    print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
