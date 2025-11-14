#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from tqdm import tqdm
from langchain_ollama import OllamaLLM

INPUT = "review_forum.jsonl"
OUTPUT = "structured_feedback.jsonl"

llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 512})

PROMPT_TEMPLATE = """
You are an expert Firefox engineer.

Given the following raw user question:

TITLE:
{title}

BODY:
{body}

Extract the following fields as a pure JSON object:

- "summary": A structured one-sentence summary using tags:
  [Problem], [Symptoms], [Environment], [Possible Category], [Problem Type]

- "category": One of:
  ["Security", "UI / Interaction", "Search", "Performance",
   "Functionality", "Configuration Issue", "Unknown"]

- "problem_type": A short phrase describing the issue type.

- "symptoms": List of concrete symptoms.

- "environment": {{
    "os": "...",
    "firefox_version": "...",
    "hardware": "..."
  }}

Return **ONLY JSON**. No explanation.
"""

def process():
    with open(INPUT, "r", encoding="utf-8") as f_in, \
         open(OUTPUT, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, desc="Processing with LLM"):
            item = json.loads(line)

            title = item.get("title", "")
            body = item.get("body", "")

            prompt = PROMPT_TEMPLATE.format(title=title, body=body)
            try:
                resp = llm.invoke(prompt)
                data = json.loads(resp)
            except Exception:
                continue

            data["id"] = item["id"]
            data["title"] = title

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Done. Saved -> {OUTPUT}")


if __name__ == "__main__":
    process()