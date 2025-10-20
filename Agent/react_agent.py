# Agent/react_agent.py
from dataclasses import dataclass
from typing import List, Optional, Union
from langchain_ollama import OllamaLLM

@dataclass
class AgentDecision:
    action: str
    result: Optional[Union[str, List[str]]] = None
    rationale: Optional[str] = None


def online_search_mock(query: str):
    print(f"🌐 Searching online for: {query}")
    return [
        "User reports indicate crash in settings due to memory leak.",
        "Recent bug reports mention fix in version 138.0.3."
    ]


def react_reasoning(user_feedback: str, collected: List[str]) -> AgentDecision:
    llm = OllamaLLM(model="llama3.2:3b", options={"num_predict": 128})

    context = f"User feedback:\n{user_feedback}\n\nCollected information:\n{collected}\n\n"

    prompt = f"""
You are a reasoning agent tasked with improving bug report generation.

Thought: (analyze whether the collected info is enough)
Action: (choose "generate" or "search" with a query)
Observation: (summarize reasoning)
Final Answer: (if generate, produce structured report; if search, explain next step)

Now reason step by step:
{context}
    """

    response = llm.invoke(prompt)
    print("\n🤖 ReAct reasoning trace:\n", response)

    thought = next((l for l in response.splitlines() if l.startswith("Thought:")), None)
    action = next((l for l in response.splitlines() if l.startswith("Action:")), None)

    if not action:
        return AgentDecision(action="none", rationale=thought)

    if "search" in action.lower():
        query = action.split("search", 1)[1].strip().strip('"')
        return AgentDecision(action="search", result=online_search_mock(query), rationale=thought)

    elif "generate" in action.lower():
        final = response.split("Final Answer:", 1)[-1].strip()
        return AgentDecision(action="generate", result=final, rationale=thought)

    return AgentDecision(action="none", rationale=thought)
