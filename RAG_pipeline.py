from langchain_ollama import OllamaLLM
import json
from dataclasses import dataclass
from typing import List, Optional, Union

#data class for agent decision
@dataclass
class AgentDecision:
    """model's decision structure"""
    action: str  # "search" / "generate" / "none"
    result: Optional[Union[str, List[str]]] = None
    rationale: Optional[str] = None  

# the lightweight LLM for judgement
judgement_llm = OllamaLLM(
    model="llama3.2:3b",
    options={"num_predict": 64}  # limit to 64 tokens for speed
)
# the powerful LLM for generation
generator_llm = OllamaLLM(
    model="gpt-oss:20b", 
    options={"num_predict": 512} 
)
user_feedback = "The app crashes when I open the settings page."
retrieved_list = [
    "Known issue: crash due to null pointer in settings menu (bug 14235).",
    "Settings page uses deprecated API in Android 13.",
    "UI rendering delay when accessing user profile tab."
]

# define the evaluation function using LLM
def evaluate_candidate(feedback: str, candidate: str) -> dict:
    """
    Use LLM to evaluate if the candidate knowledge is relevant and sufficient.
    two boolean fields: relevant, sufficient
    one string field: reason
    Returns a dict with the evaluation results.
    """
    prompt = f"""
You are an assistant that evaluates whether a retrieved text is useful for generating a bug report.

User feedback:
{feedback}

Candidate knowledge:
{candidate}

Please answer in **valid JSON** format with the following fields:
- "relevant": true or false
- "sufficient": true or false
- "reason": one short sentence explanation

Example:
{{
  "relevant": true,
  "sufficient": false,
  "reason": "It is about the same settings crash but lacks reproduction steps."
}}
    """
    response = judgement_llm.invoke(prompt)

    # parse the JSON response
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ Warning: JSON parse failed, fallback to raw text")
        result = {"relevant": False, "sufficient": False, "reason": response}

    return result
# define the generation function using LLM
def generate_bug_report(context: str) -> str:
    """
    Call Larger LLM to generate a structured bug report.
    arguments:
        context: str - combined context from user feedback and collected docs
    returns:
         bug report in JSON format
    """
    prompt = f"""
You are a software testing assistant that generates structured bug reports.

Based on the following context, create a concise and complete bug report 
in this JSON format:
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

    print("🧠 Generating structured bug report...")
    response = generator_llm.invoke(prompt)
    return response

# Agent loop
collected = [] # collected relevant but insufficient docs
for idx, doc in enumerate(retrieved_list):
    print(f"\n🔹 Checking candidate #{idx + 1}")
    result = evaluate_candidate(user_feedback, doc)

    print(f"✅ Relevant: {result['relevant']}, Sufficient: {result['sufficient']}")
    print(f"💬 Reason: {result['reason']}")

    # if both relevant and sufficient → stop and collect
    if result["relevant"] and result["sufficient"]:
        print("\n🎯 Found sufficient evidence! Proceeding to bug report generation...")
        combined_context = f"User feedback:\n{user_feedback}\n\nRelevant knowledge:\n{doc}"
        bug_report = generate_bug_report(combined_context)
        print(bug_report)
        break

    # if relevant but not sufficient → collect and continue
    elif result["relevant"]:
        collected.append(doc)


#search online if no sufficient doc found
def online_search_mock(query: str):
    print(f"🌐 Searching online for: {query}")
    # just a mock function, return some dummy results
    return [
        "User reports indicate crash in settings due to memory leak.",
        "Recent bug reports mention fix in version 138.0.3."
    ]

# ReAct Agent for adaptive search and generation
def react_reasoning(user_feedback, collected):
    """ ReAct Agent"""
    llm = OllamaLLM(model="llama3.2:3b", options={"num_predict": 128})

    context = (
        f"User feedback:\n{user_feedback}\n\n"
        f"Collected information:\n{collected}\n\n"
    )

    prompt = f"""
You are a reasoning agent tasked with improving bug report generation.
Follow this reasoning format exactly:

Thought: (analyze whether the collected information is enough)
Action: (decide what to do next — either "generate" or "search" with a query)
Observation: (summarize what you found or why you chose to generate)
Final Answer: (if generate, produce a structured bug report; if search, explain next step)

Now, reason step by step based on the input below:

{context}

Start your reasoning below.
    """

    response = llm.invoke(prompt)
    print("\n🤖 ReAct reasoning trace:\n", response)

    # actions based on response
    action_line = next((line for line in response.splitlines() if line.startswith("Action:")), None)
    if not action_line:
        return {"action": "none", "result": None}

    if "search" in action_line.lower():
        # mock search query extraction
        search_query = action_line.split("search", 1)[1].strip().strip('"')
        new_docs = online_search_mock(search_query)
        return {"action": "search", "result": new_docs}

    elif "generate" in action_line.lower():
        # mock generate bug report
        final = response.split("Final Answer:", 1)[-1].strip()
        return {"action": "generate", "result": final}

    else:
        return {"action": "none", "result": None}
    

if bug_report is None:
    if collected:
        print(" Invoking ReAct reasoning agent...")
        decision = react_reasoning(user_feedback, collected)

        if decision["action"] == "search":
            print("🔍 Online search triggered:", decision["result"])
        elif decision["action"] == "generate":
            print("📝 Generated bug report:\n", decision["result"])
        else:
            print("⚠️ Agent did not return a valid action.")
    else:
        print("No relevant information found. Cannot generate bug report.")

