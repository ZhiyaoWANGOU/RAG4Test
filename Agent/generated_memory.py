import os, json
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM


class GeneratedReportMemory:
    """
    Persistent vector memory for all generated bug reports.
    Includes LLM-based verification to ensure logical similarity.
    """

    def __init__(self, path: str = "logs/generated_vec", similarity_threshold: float = 0.8):
        """
        Initialize persistent ChromaDB collection for generated reports.
        This DB keeps embeddings on disk — no recomputation required.
        """
        os.makedirs(path, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")

        # ✅ Chroma 1.0+ (from langchain_chroma) 必须显式指定 embedding_function
        self.store = Chroma(
            collection_name="generated_reports",
            embedding_function=self.embedding,
            persist_directory=path,
        )

    # ---------- Store new report ----------
    def add_report(self, feedback: str, bug_report: str, metadata: dict | None = None):
        """
        Store a new generated bug report into the persistent vector DB.
        The embedding is computed only once during this addition.
        """
        self.store.add_texts(
            texts=[feedback],
            metadatas=[metadata or {"bug_report": bug_report}],
        )
        print("✅ Added new report to persistent memory (ChromaDB)")

    # ---------- Retrieve similar reports + LLM verification ----------
    def search_reports(self, feedback: str, top_k: int = 3, verify_llm: bool = True):
        """
        Retrieve most semantically similar feedbacks from stored memory.
        Optionally use an LLM to verify logical similarity.
        Only candidates above the similarity threshold trigger LLM verification.
        """
        # ⚠️ 注意：Chroma 返回的是“distance”（越小越相似）
        results = self.store.similarity_search_with_score(feedback, k=top_k)
        if not results:
            print("💾 No similar feedbacks found in generated memory.")
            return None

        # 计算“相似度”并筛选高相似样本
        converted = []
        for doc, distance in results:
            similarity = 1 - min(distance, 1)  # 将距离映射为[0,1]相似度
            converted.append((doc, similarity))

        filtered = [(doc, sim) for doc, sim in converted if sim >= self.similarity_threshold]
        if not filtered:
            print(f"🧩 All similarity scores below threshold ({self.similarity_threshold}). Skipping reuse.")
            return None

        print(f"🔍 Retrieved {len(filtered)} high-similarity report(s) (≥{self.similarity_threshold}).")

        # 不启用 LLM 验证时直接返回最相似的一条
        if not verify_llm:
            top_doc, sim = filtered[0]
            print(f"✅ Directly reusing top match (similarity={sim:.3f})")
            return {"report": top_doc, "similarity": sim, "rationale": "Direct similarity reuse (no LLM check)"}

        # ====== LLM Verification Layer ======
        llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 128})
        formatted = "\n\n".join([
            f"[Candidate #{i+1}]\nFeedback: {doc.page_content}\n(Similarity={sim:.3f})"
            for i, (doc, sim) in enumerate(filtered)
        ])

        prompt = f"""
You are a reasoning agent deciding whether any of the past feedback entries
describe a similar issue to the new feedback.

New feedback:
{feedback}

Candidate feedbacks from memory:
{formatted}

Output **pure JSON**:
{{
  "reuse": true/false,
  "matched_indices": [list of matching report indices, if any],
  "rationale": "short reasoning (max 1–2 sentences)"
}}
"""
        print("🧠 LLM verifying memory relevance...")
        response = llm.invoke(prompt)

        try:
            result = json.loads(response)
        except Exception:
            result = {"reuse": False, "matched_indices": [], "rationale": response}

        print("🧩 LLM validation result:", result)

        # ====== Final decision ======
        if not result.get("reuse"):
            print("❌ No reusable memory found after LLM validation.")
            return None

        matched = []
        for idx in result.get("matched_indices", []):
            if 0 <= idx - 1 < len(filtered):
                matched.append(filtered[idx - 1][0])

        if not matched:
            print("⚠️ Validation indices invalid or empty.")
            return None

        print(f"✅ Reusing memory report (feedback snippet): {matched[0].page_content[:60]}...")
        return {
            "report": matched[0],
            "similarity": filtered[0][1],
            "rationale": result.get("rationale", ""),
        }

    # ---------- Inspect stored data ----------
    def show_all(self):
        """
        Print all stored feedbacks (for debugging or inspection).
        """
        try:
            items = self.store.get()["documents"]
            print(f"📚 Currently stored feedbacks: {len(items)}")
            for i, fb in enumerate(items):
                print(f"[{i+1}] {fb[:80]}...")
        except Exception as e:
            print("⚠️ Could not read stored feedbacks:", e)