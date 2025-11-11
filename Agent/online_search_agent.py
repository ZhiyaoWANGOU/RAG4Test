# Agent/online_search_agent.py
from dataclasses import dataclass
from typing import List, Optional, Union
from langchain_ollama import OllamaLLM
from ddgs import DDGS
import json
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)


@dataclass
class AgentDecision:
    """Agent output structure."""
    action: str  # "generate" | "store"
    rationale: Optional[str] = None
    result: Optional[str] = None
    combined_context: Optional[str] = None
    raw_results_count: int = 0


def search_online(query: str, max_results: int = 8) -> List[str]:
    """Perform an online search using DuckDuckGo."""
    print(f"🌍 Searching for: {query}")
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"{r['title']}: {r['body']} ({r['href']})")
    except Exception as e:
        print("⚠️ Online search failed:", e)
    return results


def summarize_results(feedback: str, results: List[str]) -> str:
    """Summarize search results to extract likely causes or insights."""
    llm = OllamaLLM(model="llama3.2:3b", options={"num_predict": 256})

    combined_results = "\n".join(results[:8])
    prompt = f"""
You are a software analysis assistant.
Given the user's feedback and the following online findings,
summarize possible causes, patterns, or insights that may help generate a bug report.

User feedback:
{feedback}

Online findings:
{combined_results}

Summarize the relevant findings in concise technical English (2–4 sentences).
"""
    print("🧠 Summarizing search results...")
    return llm.invoke(prompt)


def judge_relevance(feedback: str, summary: str) -> AgentDecision:
    """Ask the model whether the summary is relevant and sufficient."""
    llm = OllamaLLM(model="llama3.2:3b", options={"num_predict": 128})

    prompt = f"""
You are an expert agent that decides whether the summarized online information
is relevant and sufficient to help generate a structured bug report.

User feedback:
{feedback}

Summary of online findings:
{summary}

Answer ONLY in valid JSON format with these fields:
{{
  "action": "generate" or "store",
  "rationale": "one-sentence reason why",
  "context": "if action is generate, combine feedback + summary for generation"
}}
"""
    print("🧠 Evaluating relevance of summarized findings...")
    response = llm.invoke(prompt)

    # ✅ 提前清理非 JSON 部分，提取纯 JSON 主体
    start_idx = response.find("{")
    end_idx = response.rfind("}")
    if start_idx != -1 and end_idx != -1:
        cleaned_response = response[start_idx:end_idx + 1]
    else:
        cleaned_response = response

    try:
        data = json.loads(cleaned_response)
        action = data.get("action", "store")
        rationale = data.get("rationale", "")
        combined_context = data.get("context", f"{feedback}\n\n{summary}")

        print("✅ Successfully parsed JSON decision.")
        return AgentDecision(
            action=action,
            rationale=rationale,
            combined_context=combined_context
        )

    except Exception as e:
        print(f"⚠️ JSON parse failed ({e}), fallback to 'store'")
        print("⚠️ Raw model output:\n", response)
        return AgentDecision(
            action="store",
            rationale=cleaned_response,
            combined_context=f"{feedback}\n\n{summary}"
        )


def online_search_agent(feedback: str) -> AgentDecision:
    """
    The complete online search pipeline:
    1. Generate queries
    2. Search online
    3. Summarize results
    4. Judge relevance & sufficiency
    """
    query_llm = OllamaLLM(model="llama3.2:3b", options={"num_predict": 64})

    # Step 1️⃣ generate short generic queries
    query_prompt = f"""
You are an assistant that generates web search queries to find
similar software issues to the following feedback.

Feedback:
{feedback}

Generate 3–5 concise and generic search queries (each ≤8 words)
that could help identify similar issues in other apps or systems.
Return them as a JSON list.
"""
    print("🧠 Generating search queries...")
    query_response = query_llm.invoke(query_prompt)

    try:
        queries = json.loads(query_response)
        if not isinstance(queries, list):
            raise ValueError("not a list")
    except Exception:
        print("⚠️ Query parsing failed, fallback to using raw feedback")
        queries = [feedback]

    print("🔍 Generated queries:")
    for q in queries:
        print(" -", q)

    # Step 2️⃣ search online
    all_results = []
    for q in queries:
        all_results.extend(search_online(q, max_results=5))

    all_results = list(set(all_results))
    print(f"✅ Retrieved {len(all_results)} unique results.")

    # Step 3️⃣ summarize findings
    summary = summarize_results(feedback, all_results)

    # Step 4️⃣ judge relevance
    decision = judge_relevance(feedback, summary)
    decision.raw_results_count = len(all_results)
    decision.result = summary

    # Final summary
    print("\n📊 ===== Online Search Agent Decision =====")
    print(f"Action: {decision.action}")
    print(f"Rationale: {decision.rationale}")

    return decision