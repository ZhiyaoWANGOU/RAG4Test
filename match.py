import pandas as pd
import json
import re
import os

def extract_problem_content(text):
    """
    从文本中提取 [Problem] 标记后的内容，用于生成匹配键。
    """
    if not isinstance(text, str):
        return None
    pattern = r"\[Problem\]\s*(.*?)(?=\s*(?:\[|$))"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().strip(',')
    return None

def read_file_for_matching(file_path, target_field):
    """
    读取文件，返回一个列表。
    列表包含字典：{'match_key': 提取出的键, 'raw_json': 原始的一整行JSON文本}
    """
    processed_data = []
    
    if not os.path.exists(file_path):
        print(f"[跳过] 找不到文件: {file_path}")
        return []

    print(f"正在读取 {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                # 解析 JSON 仅为了提取匹配键
                json_obj = json.loads(line)
                
                # 获取用于匹配的文本字段
                content_text = json_obj.get(target_field, "")
                
                # 生成匹配键
                key = extract_problem_content(content_text)
                
                if key:
                    processed_data.append({
                        'match_key': key,
                        'raw_json': line  # 保留原始行
                    })
            except json.JSONDecodeError:
                print(f"[警告] {file_path} 第 {i+1} 行 JSON 格式错误，已跳过。")
                continue
                
    return processed_data

def main():
    # --- 文件路径配置 ---
    # Feedback 文件列表 (你要匹配的源文件都放在这里)
    feedback_files = [
        'new_retrieved_feedbacks.jsonl',
        'retrieved_feedbacks.jsonl'
    ]
    
    # Report 文件 (被匹配的目标文件)
    reports_file = 'logs/generated_reports.jsonl'
    
    # 输出文件
    output_path = 'matched_raw_output.xlsx'

    # 1. 读取所有 Feedback 文件并合并
    all_feedback_data = []
    for f_path in feedback_files:
        # 注意：这里假设 feedback 文件的字段名都是 'user_feedback'
        file_data = read_file_for_matching(f_path, 'user_feedback')
        if file_data:
            print(f" -> 从 {f_path} 获取到 {len(file_data)} 条数据")
            all_feedback_data.extend(file_data)

    if not all_feedback_data:
        print("[错误] 没有读取到任何 Feedback 数据，程序终止。")
        return

    # 2. 读取 Reports 文件
    # 注意：这里假设 report 文件的字段名是 'feedback'
    report_data = read_file_for_matching(reports_file, 'feedback')
    if not report_data:
        print("[错误] Report 文件读取失败或为空，程序终止。")
        return

    # 3. 转换为 DataFrame
    df_feedback = pd.DataFrame(all_feedback_data)
    df_report = pd.DataFrame(report_data)

    print(f"--- 开始匹配 ---")
    print(f"Feedback 总数: {len(df_feedback)}")
    print(f"Reports 总数: {len(df_report)}")

    # 4. 执行合并 (Inner Join)
    # 这一步会根据 match_key 把两边对应的数据连起来
    # suffixes 参数用于区分两边的 raw_json 列名
    merged_df = pd.merge(df_feedback, df_report, on='match_key', how='inner', suffixes=('_Feedback', '_Report'))

    # 5. 筛选输出列 (只保留两列原始内容)
    final_df = merged_df[['raw_json_Feedback', 'raw_json_Report']]

    # 重命名列头，方便查看
    final_df.columns = ['Feedback_Json_Content', 'Report_Json_Content']

    print(f"匹配完成！共生成 {len(final_df)} 条匹配数据。")

    # 6. 导出 Excel
    try:
        final_df.to_excel(output_path, index=False)
        print(f"--- 成功 ---")
        print(f"所有匹配结果已保存至: {output_path}")
    except Exception as e:
        print(f"保存 Excel 时出错: {e}")

if __name__ == "__main__":
    main()