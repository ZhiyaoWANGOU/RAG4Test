from Agent.memory_state import MemoryState
from Agent.judgement_agent import evaluate_candidate
from Agent.generator_agent import generate_bug_report
from Agent.react_agent import react_reasoning

def main():
    # Example feedback
    user_feedback = "The app crashes when I open the settings page."
    retrieved_list = [
        "Known issue: crash due to null pointer in settings menu (bug 14235).",
        "Settings page uses deprecated API in Android 13.",
        "UI rendering delay when accessing user profile tab."
    ]

    # Step 1 Create a memory state for this feedback
    state = MemoryState(feedback=user_feedback)

    # Step 2 Main loop: evaluate each retrieved document
    for idx, doc in enumerate(retrieved_list):
        print(f"\n🔹 Checking candidate #{idx + 1}")
        result = evaluate_candidate(state.feedback, doc)
        print(result)

        if result["relevant"] and result["sufficient"]:
            print("\n Found sufficient evidence! Generating bug report...")
            context = f"{state.feedback}\n\n{doc}"
            state.set_bug_report(generate_bug_report(context))
            break
        elif result["relevant"]:
            state.add_evidence(doc)

    # Step 3 If no sufficient evidence found, use ReAct reasoning
    if not state.bug_report:
        print("\n Invoking ReAct reasoning agent...")
        decision = react_reasoning(state.feedback, state.collected)
        state.set_decision(decision.rationale)

        if decision.action == "search":
            print(" Online search triggered:")
            for d in decision.result:
                print(" -", d)

        elif decision.action == "generate":
            state.set_bug_report(decision.result)
            print(" Generated bug report:\n", decision.result)

    # Step 4 Output final state summary
    print("\n Final Memory State:")
    print(state.to_json())


if __name__ == "__main__":
    main()
