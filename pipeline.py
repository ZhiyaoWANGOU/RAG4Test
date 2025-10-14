#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline.py
修正版：使用正确的 cosine similarity 计算逻辑
"""

import json, math, csv
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# ====== 参数配置 ======
DB_PATH = "./kb_index"
LAYER1_NAME = "firefox_kb"
LAYER2_NAME = "firefox_bugs"
USER_FEEDBACK_PATH = "user_feedback.json"
OUTPUT_CSV = "pipeline_results.csv"
TOP_K = 10
THRESHOLD_LOW = 0.75   # 若相似度均低于此阈值 -> 激活 Layer3
BETA_BASE = 2.0        # 温度因子基值

# ====== 初始化客户端 ======
print("🚀 Starting cosine-similarity pipeline...\n")
client = chromadb.PersistentClient(path=DB_PATH)
collections = [c.name for c in client.list_collections()]
print("✅ Found collections:", collections)

layer1 = client.get_collection(LAYER1_NAME)
layer2 = client.get_collection(LAYER2_NAME)

# ====== 读取 feedback ======
with open(USER_FEEDBACK_PATH, "r", encoding="utf-8") as f:
    feedbacks = json.load(f)
print(f"📥 Loaded {len(feedbacks)} feedback entries.\n")

# ====== 函数定义 ======
def compute_cosine_similarity(distances):
    """Chroma 返回的 cosine 距离 = 1 - cosine_similarity"""
    return [1 - d for d in distances]

def adaptive_weights(s1, s2, alpha1=0.5, alpha2=0.5, beta_base=2.0):
    """根据相似度差异调整温度并计算权重"""
    diff = abs(s1 - s2)
    beta = beta_base * diff * 5     # 放大因子 *5 让 β 落在 0.1~1.0 区间
    w1 = alpha1 * math.exp(beta * s1)
    w2 = alpha2 * math.exp(beta * s2)
    z = w1 + w2
    return w1/z, w2/z, beta


# ====== 主循环 ======
results = []
layer3_count = 0

for fb in feedbacks:
    qtext = f"{fb['title']} {fb['summary']}"
    fid = fb["id"]

    # --- Layer1 ---
    res1 = layer1.query(query_texts=[qtext], n_results=TOP_K)
    dists1 = res1["distances"][0]
    sim1 = compute_cosine_similarity(dists1)
    s1 = max(sim1) if sim1 else 0.0

    # --- Layer2 ---
    res2 = layer2.query(query_texts=[qtext], n_results=TOP_K)
    dists2 = res2["distances"][0]
    sim2 = compute_cosine_similarity(dists2)
    s2 = max(sim2) if sim2 else 0.0

    # --- 权重计算 ---
    w1, w2, beta = adaptive_weights(s1, s2, beta_base=BETA_BASE)

    # --- Layer3 判断 ---
    need_layer3 = s1 < THRESHOLD_LOW and s2 < THRESHOLD_LOW
    if need_layer3:
        layer3_count += 1

    # --- 保存结果 ---
    results.append({
        "id": fid,
        "w1": round(w1, 4),
        "w2": round(w2, 4),
        "s1": round(s1, 4),
        "s2": round(s2, 4),
        "beta": round(beta, 4),
        "activate_layer3": int(need_layer3)
    })

# ====== 写出 CSV ======
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"✅ Done. Saved results to {OUTPUT_CSV}")
print(f"📊 Total processed: {len(results)}")
print(f"🚀 Layer3 triggered: {layer3_count} times.\n")
