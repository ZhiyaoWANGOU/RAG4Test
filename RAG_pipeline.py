from Agent.memory_state import MemoryState
from Agent.judgement_agent import evaluate_candidate
from Agent.generator_agent import generate_bug_report
from Agent.react_agent import react_reasoning, online_search_agent
from Agent.generated_memory import GeneratedReportMemory
import os, json, time

# =====================================================
# 🧩 文件路径设置
# =====================================================
DATA_PATH = "retrieved_feedbacks.jsonl"
PROGRESS_FILE = "logs/experiment_progress.json"
COUNTER_FILE = "logs/reuse_counter.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# =====================================================
# 🔄 进度追踪
# =====================================================
def load_progress():
    """读取当前实验进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("last_index", -1)
    return -1


def save_progress(idx):
    """保存当前实验进度"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_index": idx}, f)


# =====================================================
# 📈 Reuse计数逻辑（累计，不覆盖）
# =====================================================
def load_counters():
    """载入累计 reuse 计数"""
    if not os.path.exists(COUNTER_FILE):
        # ✅ 如果文件不存在，说明是第一次运行，从0开始
        return {"case_count": 0, "reuse_count": 0, "history": []}

    with open(COUNTER_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_counters(counters):
    """追加式保存 reuse 计数"""
    # 每次运行都追加一个快照，不覆盖旧的历史
    counters["history"].append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": counters["case_count"],
        "reuse_count": counters["reuse_count"],
        "reuse_rate": round(counters["reuse_count"] / max(1, counters["case_count"]), 4)
    })
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        json.dump(counters, f, ensure_ascii=False, indent=2)


# =====================================================
# 🧠 主Pipeline逻辑
# =====================================================
def main():
    # ========== Step 0: 载入数据 ==========
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f if line.strip()]

    last_index = load_progress()
    next_index = last_index + 1

    if next_index >= len(data_lines):
        print("✅ All feedbacks processed. Nothing left.")
        return

    entry = data_lines[next_index]
    user_feedback = entry.get("user_feedback", "")
    retrieved_list = entry.get("retrieved_list", [])
    feedback_id = entry.get("id", f"case_{next_index}")
    save_progress(next_index)

    print(f"\n===== 🧩 Running Case #{next_index + 1} | ID: {feedback_id} =====")

    # ========== Step 1: 载入计数器 ==========
    counters = load_counters()
    counters["case_count"] += 1

    # ========== Step 2: 尝试 Reuse ==========
    print("🧠 Checking if feedback matches any existing generated reports...")
    memory = GeneratedReportMemory()
    reuse = memory.search_reports(user_feedback, top_k=3, verify_llm=True)

    if reuse:
        counters["reuse_count"] += 1
        doc = reuse["report"]
        sim_feedback = doc.page_content
        sim_report = doc.metadata.get("bug_report", "")
        print(f"\n🔁 Found reusable report: {sim_feedback[:80]}...")

        state = MemoryState(feedback=user_feedback)
        state.set_decision("Reused memory report (similarity reuse + LLM verified)")
        state.set_bug_report(sim_report)
        print("Reused bug report:\n", sim_report)

        print("\nFinal Memory State:")
        print(state.to_json())

        # ✅ 保存计数（追加方式）
        save_counters(counters)

        print(f"📊 Reuse success rate so far: {counters['reuse_count']}/{counters['case_count']} "
              f"({counters['reuse_count'] / counters['case_count']:.2%})")
        return  # ✅ Reuse成功则停止

    print("❌ No reusable memory found. Proceeding with normal reasoning...\n")

    # ========== Step 3: 创建 MemoryState ==========
    state = MemoryState(feedback=user_feedback)

    # ========== Step 4: 评估 retrieved 文档 ==========
    for idx, doc in enumerate(retrieved_list):
        print(f"\n🔎 Checking candidate #{idx + 1}")
        result = evaluate_candidate(state.feedback, doc)
        print(result)

        if result["relevant"] and result["sufficient"]:
            print("\n✅ Found sufficient evidence! Generating bug report...")
            context = f"{state.feedback}\n\n{doc}"
            structured_report = generate_bug_report(context)
            state.set_bug_report(structured_report)
            state.set_decision("Action: Generate (sufficient evidence found locally, reasoning skipped)")

            # ✅ 存入向量数据库与日志
            memory.add_report(state.feedback, state.bug_report)
            with open("logs/generated_reports.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "feedback": state.feedback,
                    "collected": state.collected,
                    "decision": state.decision,
                    "bug_report": state.bug_report
                }, ensure_ascii=False) + "\n")

            print("📝 Generated report saved to logs/generated_reports.jsonl")
            save_counters(counters)
            break

        elif result["relevant"]:
            state.add_evidence(doc)

    # ========== Step 5: 若无结果，调用推理代理 ==========
    if not state.bug_report:
        print("\n🧠 Invoking ReAct reasoning agent...")
        decision = react_reasoning(state.feedback, state.collected)
        print(">>> ReAct returned decision:", decision.action)
        state.set_decision(decision.rationale)

        if decision.action == "search":
            print("🌍 Online search triggered...")
            search_decision = online_search_agent(state.feedback)
            state.set_decision(f"{decision.rationale}\n{search_decision.rationale}")

            if search_decision.action == "generate":
                print("\n✅ Online search provided sufficient info. Generating bug report...")
                collected_text = "\n".join(state.collected) if state.collected else ""
                combined_context = (
                    f"User feedback:\n{state.feedback}\n\n"
                    f"Previously collected information:\n{collected_text}\n\n"
                    f"Online search summary:\n{search_decision.combined_context}"
                )
                structured_report = generate_bug_report(combined_context)
                state.set_bug_report(structured_report)
                print(state.bug_report)

                memory.add_report(state.feedback, state.bug_report)
                with open("logs/generated_reports.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "feedback": state.feedback,
                        "collected": state.collected,
                        "decision": state.decision,
                        "bug_report": state.bug_report
                    }, ensure_ascii=False) + "\n")

                print("📝 Generated report saved to logs/generated_reports.jsonl")

            elif search_decision.action == "store":
                stored_entry = {
                    "feedback": state.feedback,
                    "collected": state.collected,
                    "rationale": search_decision.rationale,
                    "summary": search_decision.combined_context,
                }
                with open("logs/stored_feedback.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(stored_entry, ensure_ascii=False) + "\n")
                print("📝 Stored feedback saved to logs/stored_feedback.jsonl")

        elif decision.action == "generate":
            print("🧩 Generating bug report directly from reasoning output...")
            structured_report = generate_bug_report(decision.result)
            state.set_bug_report(structured_report)
            print("Generated bug report:\n", decision.result)

            memory.add_report(state.feedback, state.bug_report)
            with open("logs/generated_reports.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "feedback": state.feedback,
                    "collected": state.collected,
                    "decision": state.decision,
                    "bug_report": state.bug_report
                }, ensure_ascii=False) + "\n")

            print("📝 Generated report saved to logs/generated_reports.jsonl")

        elif decision.action == "store":
            stored_entry = {
                "feedback": state.feedback,
                "collected": state.collected,
                "rationale": decision.rationale,
                "summary": getattr(decision, "combined_context", ""),
            }
            with open("logs/stored_feedback.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(stored_entry, ensure_ascii=False) + "\n")
            print("📝 Stored feedback saved to logs/stored_feedback.jsonl")

    # ========== Step 6: 保存结果与统计 ==========
    save_counters(counters)
    print("\nFinal Memory State:")
    print(state.to_json())

    print(f"\n📊 Reuse Stats Summary:")
    print(f"  - Total cases processed: {counters['case_count']}")
    print(f"  - Total reuses: {counters['reuse_count']}")
    print(f"  - Reuse rate: {counters['reuse_count'] / counters['case_count']:.2%}")


if __name__ == "__main__":
    # ============ 自动运行全部反馈 ============
    DATA_PATH = "retrieved_feedbacks.jsonl"
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    print(f"📥 Loaded {len(lines)} feedback entries.")
    for i, entry in enumerate(lines, start=1):
        print(f"\n================= 🧩 Starting Case #{i}/{len(lines)} =================")
        user_feedback = entry.get("user_feedback", "")
        retrieved_list = entry.get("retrieved_list", [])
        try:
            main()
        except Exception as e:
            print(f"⚠️ Case #{i} failed: {e}")