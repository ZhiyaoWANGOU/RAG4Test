#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_support_questions.py

从 Mozilla Support Forum 抓取 Firefox 问题帖：
  https://support.mozilla.org/en-US/questions/firefox?show=all&page=N

每页约 20 条，默认抓取前 10 页（可调）。
每条数据访问详情页提取正文。

输出：
  support_forum.jsonl
示例字段：
{
  "id": "1539261",
  "title": "impossibilité d'ouvrir Firefox",
  "url": "https://support.mozilla.org/en-US/questions/1539261",
  "body": "Bonjour. Je n'arrive pas à ouvrir Firefox dont je me sers depuis 15 ans...",
  "lang": "fr"
}
"""

import requests
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
import time, json, os, argparse, re

DetectorFactory.seed = 0  # 保证语言检测可复现

BASE = "https://support.mozilla.org"
LIST_URL = f"{BASE}/en-US/questions/firefox?show=all&page={{}}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataCollector/1.0)"}


def fetch_list_page(page: int):
    """抓取列表页并解析出每条问题的id、title、url"""
    url = LIST_URL.format(page)
    print(f"[list] Fetching page {page}: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items = []
    for art in soup.select("section.forum--question-list article.forum--question-item"):
        a = art.select_one("h2 a")
        if not a:
            continue
        qid = art.get("id", "").split("-")[-1]
        href = a.get("href", "")
        title = a.get_text(strip=True)
        full_url = BASE + href
        items.append({"id": qid, "title": title, "url": full_url})
    print(f"  -> {len(items)} items found")
    return items


def fetch_question_body(url: str) -> str:
    """进入详情页提取正文（div.main-content 内的 p）"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        main_div = soup.select_one("div.main-content")
        if not main_div:
            return ""
        paras = [p.get_text(" ", strip=True) for p in main_div.select("p")]
        body = " ".join(paras)
        body = re.sub(r"\s+", " ", body)
        return body.strip()
    except Exception as e:
        print(f"[warn] fail to fetch {url}: {e}")
        return ""


def detect_lang_safe(text: str) -> str:
    """安全语言检测"""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def crawl_forum(pages=10, delay=1.5, out_path="support_forum.jsonl"):
    all_items = []
    with open(out_path, "w", encoding="utf-8") as out:
        for p in range(1, pages + 1):
            lst = fetch_list_page(p)
            for item in lst:
                body = fetch_question_body(item["url"])
                if not body:
                    continue
                lang = detect_lang_safe(body)
                item["body"] = body
                item["lang"] = lang
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                all_items.append(item)
                print(f"    [+] {item['id']} ({lang}) {item['title'][:60]}...")
                time.sleep(delay)
            time.sleep(delay * 2)
    print(f"\n✅ Done. {len(all_items)} questions saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=10, help="抓取页数 (每页约20条)")
    ap.add_argument("--delay", type=float, default=1.5, help="请求间隔 (秒)")
    ap.add_argument("--out", default="support_forum.jsonl", help="输出文件")
    args = ap.parse_args()

    crawl_forum(pages=args.pages, delay=args.delay, out_path=args.out)
