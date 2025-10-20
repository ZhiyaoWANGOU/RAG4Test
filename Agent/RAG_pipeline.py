from Agent.judgement_agent import evaluate_candidate
from Agent.generator_agent import generate_bug_report
from Agent.react_agent import react_reasoning, AgentDecision
from memory_module import SemanticMemory

# main logic
user_feedback = "The app crashes when I open the settings page."
retrieved_list = [...]

collected, bug_report = [], None

for idx, doc in enumerate(retrieved_list):
    print(f"\n🔹 Checking candidate #{idx + 1}")
    result = evaluate_candidate(user_feedback, doc)
    print(result)

    if result["relevant"] and result["sufficient"]:
        bug_report = generate_bug_report(f"{user_feedback}\n\n{doc}")
        break
    elif result["relevant"]:
        collected.append(doc)

if not bug_report:
    print("\nInvoking ReAct agent...")
    decision = react_reasoning(user_feedback, collected)
