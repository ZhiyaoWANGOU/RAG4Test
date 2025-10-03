#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_bug.py

Modes
-----
1) 写入向量库（Ollama + nomic-embed-text -> ChromaDB）：
   python embed_bug.py --mode embed --jsonl bugs_corpus.jsonl --db ./kb_index --col firefox_bugs

2) 本地检索：
   python embed_bug.py --mode query --db ./kb_index --col firefox_bugs --q "search keywords"
"""

from __future__ import annotations
import argparse, json, os, uuid
from typing import List, Dict


# ---------- mode=embed ----------
def embed_to_chroma(jsonl_path: str, db_path: str, col_name: str,
                    model_name: str = "nomic-embed-text",
                    batch_size: int = 128):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=db_path)
    ollama_embed = embedding_functions.OllamaEmbeddingFunction(
        model_name=model_name,
    )
    col = client.get_or_create_collection(name=col_name, embedding_function=ollama_embed)

    # 读入所有行
    rows: List[Dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    print(f"[embed] total bugs: {total}")

    # 批量写入
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]

        col.add(
            ids=[str(r["id"]) for r in batch],
            documents=[f"[{r['comp']}] {r['summary']}" for r in batch],
            metadatas=[{
                "bug_id": r["id"],
                "summary": r["summary"],
                "comp": r["comp"],
                "type": r["type"],
                "product": r["product"]
            } for r in batch]
        )
        print(f"[embed] inserted {min(i+batch_size, total)}/{total}")

    print(f"✅ ChromaDB OK -> {db_path} / {col_name}")


# ---------- mode=query ----------
def query_demo(db_path: str, col_name: str, q: str, k: int = 5):
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_or_create_collection(name=col_name)  # embedding_function已持久化

    res = col.query(query_texts=[q], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or res.get("embeddings", [[]])

    print("\n=== Top Results ===")
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        dist = dists[i-1] if i-1 < len(dists) else None
        print(f"[{i}] Bug {meta.get('bug_id')} | {meta.get('comp')}")
        print(f"    Type: {meta.get('type')}")
        if dist is not None:
            print(f"    distance: {dist:.4f}")
        print(f"    summary: {meta.get('summary')}")
        print(f"    text: {doc[:300]}...\n")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["embed", "query"])
    ap.add_argument("--jsonl", help="bugs corpus JSONL 路径")
    ap.add_argument("--db", help="ChromaDB 路径（目录）", default="./kb_index")
    ap.add_argument("--col", help="ChromaDB collection 名称", default="firefox_bugs")
    ap.add_argument("--q", help="查询语句（mode=query）")
    args = ap.parse_args()

    if args.mode == "embed":
        if not args.jsonl:
            ap.error("--jsonl required for mode=embed")
        embed_to_chroma(args.jsonl, args.db, args.col)

    elif args.mode == "query":
        if not args.q:
            ap.error("--q required for mode=query")
        query_demo(args.db, args.col, args.q)


if __name__ == "__main__":
    main()
