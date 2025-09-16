#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intent_extraction.py

目标：
- 调用本地 Ollama (llama3.2)，输入用户评论
- 输出一个纯字符串总结：功能模块 / 问题类型 / 环境
"""

import argparse
import os
import sys
import requests

OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROMPT_TEMPLATE = """
You are analyzing a user bug report comment.

Task:
From the comment, extract and summarize in English:
1. The functional module where the issue happens
2. The type of issue (e.g., performance, crash, incorrect behavior, missing UI)
3. The environment (platform, OS version, app version, etc.)

Output format (plain text, no JSON):
Feature Module: ...
Issue Type: ...
Environment: ...

User comment:
<<<{comment}>>>
""".strip()


def check_ollama(model: str):
    r = requests.get(f"{OLLAMA}/api/tags", timeout=10, proxies={"http": None, "https": None})
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    if model not in models:
        print(f"[error] model '{model}' not found. available={models}", file=sys.stderr)
        sys.exit(1)


def call_generate(model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.2, "num_predict": 200}}
    r = requests.post(f"{OLLAMA}/api/generate",
                      json=payload,
                      timeout=60,
                      proxies={"http": None, "https": None})
    r.raise_for_status()
    return r.json().get("response", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comment", required=True, help="User comment text")
    ap.add_argument("--model", default="llama3.2:latest", help="Ollama model tag")
    args = ap.parse_args()

    check_ollama(args.model)

    prompt = PROMPT_TEMPLATE.format(comment=args.comment)
    result = call_generate(args.model, prompt)
    print(result.strip())


if __name__ == "__main__":
    main()
