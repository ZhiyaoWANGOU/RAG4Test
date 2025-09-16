#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embed_kb.py

Modes
-----
1) 生成语料 JSONL：
   python embed_kb.py --mode corpus --kb-dir kb_json --out kb_corpus.jsonl

2) 写入向量库（Ollama + nomic-embed-text -> ChromaDB）：
   python embed_kb.py --mode embed --jsonl kb_corpus.jsonl --db ./kb_index --col firefox_kb

3) 本地检索（可选）：
   python embed_kb.py --mode query --db ./kb_index --col firefox_kb --q "restore profile folder"
"""

from __future__ import annotations
import argparse, glob, json, os, uuid, math
from typing import List, Dict

# ---------- 可选：轻量清洗 ----------
def _clean_text(s: str) -> str:
    s = (s or "").replace("\xa0", " ").strip()
    return " ".join(s.split())

# ---------- mode=corpus ----------
def build_corpus(kb_dir: str, out_jsonl: str, join_title: bool = True):
    files = sorted(glob.glob(os.path.join(kb_dir, "*.json")))
    n_out = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for fp in files:
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except Exception as e:
                print(f"[warn] skip {fp}: {e}")
                continue
            url   = data.get("url", "")
            title = data.get("title", "")
            for sec in data.get("sections", []):
                sec_title = sec.get("title", "")
                body = _clean_text(sec.get("text", ""))
                if not body:
                    continue
                # 向量文本 = 可选地把标题并入正文（更好召回）
                doc = f"{sec_title}\n\n{body}" if join_title and sec_title else body
                item = {
                    "id": str(uuid.uuid4()),
                    "url": url,
                    "title": title,
                    "section_title": sec_title,
                    "level": sec.get("level", None),
                    "text": doc
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"✅ wrote {n_out} lines to {out_jsonl}")

# ---------- mode=embed ----------
def embed_to_chroma(jsonl_path: str, db_path: str, col_name: str,
                    ollama_base: str = "http://localhost:11434",
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

    # 批量写入
    total = len(rows)
    print(f"[embed] total sections: {total}")
    for i in range(0, total, batch_size):
        batch = rows[i:i+batch_size]
        col.add(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[{
                "url": r["url"],
                "title": r["title"],
                "section": r["section_title"],
                "level": r.get("level")
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
        print(f"[{i}] {meta.get('title')}  |  {meta.get('section')}")
        if dist is not None:
            print(f"    distance: {dist:.4f}")
        print(f"    url: {meta.get('url')}")
        preview = doc[:300].replace("\n", " ")
        print(f"    text: {preview}...\n")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["corpus","embed","query"])
    ap.add_argument("--kb-dir", help="目录，包含 kb_builder 的 JSON 输出（doc_*.json）")
    ap.add_argument("--out", help="corpus 输出 JSONL 文件")
    ap.add_argument("--jsonl", help="corpus JSONL 路径")
    ap.add_argument("--db", help="ChromaDB 路径（目录）", default="./kb_index")
    ap.add_argument("--col", help="ChromaDB collection 名称", default="firefox_kb")
    ap.add_argument("--q", help="查询语句（mode=query）")
    args = ap.parse_args()

    if args.mode == "corpus":
        if not args.kb_dir or not args.out:
            ap.error("--kb-dir and --out required for mode=corpus")
        build_corpus(args.kb_dir, args.out)

    elif args.mode == "embed":
        if not args.jsonl:
            ap.error("--jsonl required for mode=embed")
        embed_to_chroma(args.jsonl, args.db, args.col)

    elif args.mode == "query":
        if not args.q:
            ap.error("--q required for mode=query")
        query_demo(args.db, args.col, args.q)

if __name__ == "__main__":
    main()
