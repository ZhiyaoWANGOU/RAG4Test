from Agent.memory_state import MemoryState
from Agent.judgement_agent import evaluate_candidate
from Agent.generator_agent import generate_bug_report
from Agent.react_agent import react_reasoning
from Agent.react_agent import react_reasoning, online_search_agent
def main():
    # Example feedback
    user_feedback = "The app crashes when I open the settings page."
    retrieved_list = [
        "Good morning.",
        "Good afternoon.",
        "I wanna eat mcdonalds."
    ]

    # Step 1 Create a memory state for this feedback
    state = MemoryState(feedback=user_feedback)

    # Step 2 Main loop: evaluate each retrieved document
    for idx, doc in enumerate(retrieved_list):
        print(f"\n Checking candidate #{idx + 1}")
        result = evaluate_candidate(state.feedback, doc)
        print(result)

        if result["relevant"] and result["sufficient"]:
            print("\n Found sufficient evidence! Generating bug report...")
            context = f"{state.feedback}\n\n{doc}"
            state.set_bug_report(generate_bug_report(context))
            state.set_decision("Action: Generate (sufficient evidence found locally, reasoning skipped)")
            break
        elif result["relevant"]:
            state.add_evidence(doc)

    # Step 3 If no sufficient evidence found → invoke reasoning agent
    if not state.bug_report:
        print("\nInvoking ReAct reasoning agent...")
        decision = react_reasoning(state.feedback, state.collected)
        print(">>> ReAct returned decision:", decision.action)
        state.set_decision(decision.rationale)

        # 3-A：Reasoning 决定要联网搜索
        if decision.action == "search":
            print("Online search triggered...")
            search_decision = online_search_agent(state.feedback)
            state.set_decision(decision.rationale)
            if search_decision.action == "generate":
                print("\nOnline search provided sufficient info. Generating bug report...")
                collected_text = "\n".join(state.collected) if state.collected else ""
                combined_context = (
                    f"User feedback:\n{state.feedback}\n\n"
                    f"Previously collected information:\n{collected_text}\n\n"
                    f"Online search summary:\n{search_decision.combined_context}"
                )
                state.set_bug_report(generate_bug_report(combined_context))
                print(state.bug_report)

            elif search_decision.action == "store":
                print("\nOnline search insufficient. Feedback stored for later processing.")
            else:
                print("\nOnline search did not return a valid decision.")

        # 3-B：Reasoning 直接生成报告
        elif decision.action == "generate":
            print("Generating bug report directly from reasoning output...")
            state.set_bug_report(decision.result)
            print("Generated bug report:\n", decision.result)

        # 3-C：Reasoning 无法决策
        else:
            print("No actionable decision from ReAct reasoning agent.")

    # Step 4️⃣ Output final state summary
    print("\nFinal Memory State:")
    print(state.to_json())

if __name__ == "__main__":
    main()
