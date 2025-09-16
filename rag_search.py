#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Search Pipeline (Priority: Help Center > Bug Tracker > Q&A > Blogs)
- Free, no API keys required.
- Input: user comment (any language)
- Output: structured intent + prioritized web evidence + ready-to-use LLM prompt
"""
try:
    from ddgs import DDGS          # 新包名
except ImportError:
    from duckduckgo_search import DDGS  # 旧包名
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from langdetect import detect
from rapidfuzz import fuzz
import trafilatura, re, time, hashlib, argparse, json, sys, os
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings(
    "ignore",
    message="This package (`duckduckgo_search`) has been renamed to `ddgs`!"
)
# ----------------------- Config: priority domain sets ----------------------- #
# 你可以替换为目标产品对应的域名；默认以 Firefox 为例
DEFAULT_DOMAINS = {
    "help": [
        "support.mozilla.org",
        "www.mozilla.org"
    ],
    "bugs": [
        "bugzilla.mozilla.org",
        "github.com/mozilla"
    ],
    "qa": [
        "stackoverflow.com",
        "superuser.com",
        "askubuntu.com",
        "reddit.com/r/firefox"
    ],
    "blogs": [
        "medium.com",
        "dev.to",
        "gist.github.com",
        "superuser.com/blog"
    ]
}

# 每一层优先取多少条（可按需调参）
TIER_QUOTA = {
    "help": 6,
    "bugs": 5,
    "qa": 4,
    "blogs": 3
}

MIN_ARTICLE_LEN = 400     # 过滤太短的正文
MAX_PER_DOMAIN = 2        # 单域上限，防止一边倒
FUZZY_DUP_TITLE = 92      # 标题去重阈值（0-100）
TOPK_FINAL = 8            # 最终证据数量（总和）

# ---------------------------- Intent extraction ---------------------------- #
@dataclass
class Intent:
    feature: str
    symptom: str
    qualifiers: List[str]
    lang: str
    raw_query: str

OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

def _http_post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    import urllib.request, urllib.error
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {e}")

def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
    resp = _http_post_json(OLLAMA_CHAT_URL, payload)
    # Ollama /api/chat non-stream response typically has a 'message' field
    msg = resp.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if not content:
        # Fallback: some variants return 'messages' or 'response'
        content = resp.get("response") or ""
    return str(content)

def extract_intent_llm(comment: str, model: str = "llama3") -> Intent:
    system = (
        "You are an assistant that extracts structured intent from a user's comment about software issues. "
        "Return a strict JSON object with keys: feature, symptom, qualifiers (array), lang (ISO-639-1), raw_query. "
        "Keep it concise; guess 'General' if not clear."
    )
    user = f"""
Comment:
"""
{comment.strip()}
"""

Output JSON keys and example:
{{
  "feature": "Search",
  "symptom": "Performance",
  "qualifiers": ["windows 11", "firefox 127"],
  "lang": "en",
  "raw_query": "{comment.strip()[:200]}"
}}
Only output JSON.
"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    content = ollama_chat(model, messages, temperature=0.0)
    def _parse_json(s: str) -> dict:
        try:
            return json.loads(s)
        except Exception:
            # Try to extract the first JSON object substring
            m = re.search(r"\{.*\}", s, flags=re.S)
            if m:
                return json.loads(m.group(0))
            raise
    try:
        obj = _parse_json(content)
        feature = str(obj.get("feature", "General")).strip() or "General"
        symptom = str(obj.get("symptom", "General")).strip() or "General"
        quals = obj.get("qualifiers", [])
        if not isinstance(quals, list):
            quals = [str(quals)] if quals else []
        quals = [str(x).strip() for x in quals if str(x).strip()]
        lang = str(obj.get("lang", "en")).strip() or "en"
        raw_query = str(obj.get("raw_query", comment)).strip() or comment
    except Exception:
        # Fallback if parsing fails
        feature, symptom, lang, quals = "General", "General", "en", []
        raw_query = re.sub(r"\s+", " ", comment).strip()[:240]
    return Intent(feature=feature, symptom=symptom, qualifiers=quals, lang=lang, raw_query=raw_query)

# ---------------------------------- CLI ----------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Intent extractor using local Ollama LLM")
    ap.add_argument("--comment", required=True, help="User comment text")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3"), help="Ollama model name (e.g., llama3, llama3:8b)")
    args = ap.parse_args()

    intent = extract_intent_llm(args.comment, model=args.model)
    print(json.dumps({"intent": asdict(intent)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
