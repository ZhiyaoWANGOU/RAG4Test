#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
retrieval_pipeline.py
双层置信度检索（仅使用 summary 查询）→ 输出 agent 输入格式
"""

import json, math
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# ====== 参数配置 ======
DB_PATH = "./kb_index"
LAYER1_NAME = "firefox_kb"
LAYER2_NAME = "firefox_bugs"
USER_FEEDBACK_PATH = "user_feedback.json"
OUTPUT_PATH = "retrieved_feedbacks.jsonl"
TOP_K = 10               # 每层先取前10
FINAL_TOP_N = 5          # 最终取前5个融合结果
THRESHOLD_LOW = 0.75
BETA_BASE = 2.0

# ====== 初始化客户端 ======
print("🚀 Starting 2-layer retrieval (summary-only)...")
client = chromadb.PersistentClient(path=DB_PATH)
layer1 = client.get_collection(LAYER1_NAME)
layer2 = client.get_collection(LAYER2_NAME)

# ====== 读取 feedback ======
with open(USER_FEEDBACK_PATH, "r", encoding="utf-8") as f:
    feedbacks = json.load(f)
print(f"📥 Loaded {len(feedbacks)} feedback entries.\n")

# ====== 工具函数 ======
def compute_cosine_similarity(distances):
    """Chroma 返回的 cosine 距离 = 1 - cosine_similarity"""
    return [1 - d for d in distances]

def adaptive_weights(s1, s2, alpha1=0.5, alpha2=0.5, beta_base=2.0):
    """根据相似度差异调整温度并计算权重"""
    diff = abs(s1 - s2)
    beta = beta_base * diff * 5
    w1 = alpha1 * math.exp(beta * s1)
    w2 = alpha2 * math.exp(beta * s2)
    z = w1 + w2
    return w1/z, w2/z, beta

# ====== 主循环 ======
with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for fb in feedbacks:
        fid = fb["id"]
        qtext = fb["summary"].strip()
        if not qtext:
            continue

        # --- Layer1 ---
        res1 = layer1.query(query_texts=[qtext], n_results=TOP_K)
        docs1 = res1["documents"][0]
        sims1 = compute_cosine_similarity(res1["distances"][0])
        s1 = max(sims1) if sims1 else 0.0

        # --- Layer2 ---
        res2 = layer2.query(query_texts=[qtext], n_results=TOP_K)
        docs2 = res2["documents"][0]
        sims2 = compute_cosine_similarity(res2["distances"][0])
        s2 = max(sims2) if sims2 else 0.0

        # --- 动态权重计算 ---
        w1, w2, beta = adaptive_weights(s1, s2, beta_base=BETA_BASE)

        # --- 融合结果 + 加权得分排序 ---
        combined = []
        for doc, sim in zip(docs1, sims1):
            combined.append({"text": doc, "score": w1 * sim})
        for doc, sim in zip(docs2, sims2):
            combined.append({"text": doc, "score": w2 * sim})

        combined = sorted(combined, key=lambda x: x["score"], reverse=True)
        retrieved_list = [x["text"].strip() for x in combined[:FINAL_TOP_N] if x["text"].strip()]

        # --- 输出格式 ---
        fid = str(fb["id"])
        sample = {
            "id": fid,
            "user_feedback": qtext,
            "retrieved_list": retrieved_list,
            "weights": {"w1": round(w1, 4), "w2": round(w2, 4), "beta": round(beta, 4)},
        }

        out.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"✅ Saved retrieval results to {OUTPUT_PATH}")