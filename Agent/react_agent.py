# Agent/react_agent.py
from dataclasses import dataclass
from typing import List, Optional, Union
from langchain_ollama import OllamaLLM
from ddgs import DDGS  
import json
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)


# ===================== 基础数据结构 =====================
@dataclass
class AgentDecision:
    """Final decision for ReAct reasoning or search stage."""
    action: str                      # "generate" | "search" | "store" | "none"
    result: Optional[Union[str, List[str]]] = None
    rationale: Optional[str] = None
    combined_context: Optional[str] = None
    raw_results_count: int = 0


# ===================== ReAct Reasoning =====================
def react_reasoning(user_feedback: str, collected: List[str]) -> AgentDecision:
    """
    Reasoning agent that decides whether to:
    - generate a bug report directly, or
    - perform an online search (invokes online_search_agent)
    """
    from Agent.react_agent import online_search_agent  # 避免循环导入
    llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 128})

    context = (
        f"User feedback:\n{user_feedback}\n\n"
        f"Collected information:\n{collected}\n\n"
    )

    prompt = f"""
You are a reasoning agent assisting in automated bug report generation.

Your task:
1. Analyze whether the collected information is sufficient.
2. Decide whether to "generate" a report directly or "search" for more data.
3. Always include the word "generate" or "search" explicitly when making a decision.

Now reason step by step:
{context}
    """

    response = llm.invoke(prompt)
    print("\n🧩 ReAct reasoning trace:\n", response)

    lower_resp = response.lower()

    # Case 1: trigger online search
    if "search" in lower_resp:
        print("\n🧠 ReAct decided to perform online search...")
        search_decision = online_search_agent(user_feedback)
        print(f"🔎 Online Search Decision: {search_decision.action}")

        if search_decision.action == "generate":
            return AgentDecision(
                action="generate",
                result=search_decision.combined_context,
                rationale=f"{response}\n(Used online search summary)",
                combined_context=search_decision.combined_context,
                raw_results_count=search_decision.raw_results_count
            )
        else:
            return AgentDecision(
                action="store",
                rationale=f"{response}\n(Online search insufficient)",
                combined_context=search_decision.combined_context,
                raw_results_count=search_decision.raw_results_count
            )

    # Case 2: directly generate
    elif "generate" in lower_resp:
        final = response.split("Final Answer:", 1)[-1].strip() if "final answer:" in lower_resp else response.strip()
        return AgentDecision(
            action="generate",
            result=final,
            rationale=response
        )

    # Fallback: no recognizable action
    print("⚠️ ReAct reasoning produced no actionable decision.")
    return AgentDecision(action="none", rationale=response)


# ===================== Online Search Agent =====================
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
    if not results:
        return "No relevant online results found."

    llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 128})
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
    llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 128})

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
    print("\n===== 🧭 Entering online_search_agent() =====")
    query_llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 128})

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

    # Step 2: Online search
    all_results = []
    for q in queries:
        all_results.extend(search_online(q, max_results=5))

    all_results = list(set(all_results))
    print(f"✅ Retrieved {len(all_results)} unique results.")

    # Step 3: Summarize results
    summary = summarize_results(feedback, all_results)

    # Step 4: Judge relevance
    decision = judge_relevance(feedback, summary)
    decision.raw_results_count = len(all_results)
    decision.result = summary

    # Final summary
    print("\n📊 ===== Online Search Agent Decision =====")
    print(f"Action: {decision.action}")
    print(f"Rationale: {decision.rationale}")
    print(f"Raw results retrieved: {decision.raw_results_count}")

    return decision
