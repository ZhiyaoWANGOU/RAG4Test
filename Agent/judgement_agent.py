# Agent/judgement_agent.py
from langchain_ollama import OllamaLLM
import json

# Initialize lightweight model
judgement_llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 128})

def evaluate_candidate(feedback: str, candidate: str) -> dict:
    """
    Evaluate if candidate knowledge is relevant and sufficient for bug report generation.
    Returns: dict with fields {relevant, sufficient, reason}.
    """
    prompt = f"""
You are an assistant that evaluates whether a retrieved text is useful for generating a bug report.

User feedback:
{feedback}

Candidate knowledge:
{candidate}

Answer in JSON with fields:
{{"relevant": bool, "sufficient": bool, "reason": "one short explanation"}}
    """

    response = judgement_llm.invoke(prompt)
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {"relevant": False, "sufficient": False, "reason": response}

    return result
