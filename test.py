#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_random_similarity.py
测试无关语句在 Firefox KB/Bug collection 中的相似度分布
"""

import numpy as np
import chromadb

DB_PATH = "./kb_index"
LAYER1 = "firefox_kb"
LAYER2 = "firefox_bugs"

queries = [
    "how to cook pasta al dente",
    "best way to train a dog",
    "stock prices falling sharply today",
    "famous paintings of the renaissance",
    "soccer match results last night",
    "the weather forecast in tokyo tomorrow",
    "learn to play guitar chords easily",
    "medical symptoms of vitamin D deficiency",
    "how to install windows 11 from USB",
    "python script to send email automatically"
]

print("🚀 Testing unrelated query similarity...\n")

client = chromadb.PersistentClient(path=DB_PATH)
kb = client.get_collection(LAYER1)
bugs = client.get_collection(LAYER2)

def cosine_from_distance(d):
    return 1 - d

all_sims = []

for q in queries:
    res1 = kb.query(query_texts=[q], n_results=1)
    res2 = bugs.query(query_texts=[q], n_results=1)
    d1 = res1["distances"][0][0]
    d2 = res2["distances"][0][0]
    s1, s2 = cosine_from_distance(d1), cosine_from_distance(d2)
    all_sims += [s1, s2]
    doc1 = res1["documents"][0][0][:120].replace("\n", " ")
    doc2 = res2["documents"][0][0][:120].replace("\n", " ")
    print(f"🔍 Query: {q}")
    print(f"  Layer1 sim={s1:.4f}, dist={d1:.4f}, preview: {doc1}")
    print(f"  Layer2 sim={s2:.4f}, dist={d2:.4f}, preview: {doc2}\n")

all_sims = np.array(all_sims)
print("=====================================================")
print(f"Mean similarity: {all_sims.mean():.4f}")
print(f"Std deviation  : {all_sims.std():.4f}")
print(f"Min similarity : {all_sims.min():.4f}")
print(f"Max similarity : {all_sims.max():.4f}")
print("=====================================================")
