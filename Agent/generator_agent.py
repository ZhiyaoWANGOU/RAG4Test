# Agent/generator_agent.py
from langchain_ollama import OllamaLLM

generator_llm = OllamaLLM(model="gpt-oss:20b", options={"num_predict": 512})

def generate_bug_report(context: str) -> str:
    """
    Generate structured bug report in JSON format from context.
    """
    prompt = f"""
You are a software testing assistant that generates structured bug reports.

Based on the following context, output JSON:
{{
  "title": "...",
  "steps_to_reproduce": ["..."],
  "expected_result": "...",
  "actual_result": "...",
  "possible_cause": "...",
  "severity": "low | medium | high"
}}

Context:
{context}
    """
    print(" Generating structured bug report...")
    return generator_llm.invoke(prompt)
