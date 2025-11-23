import pandas as pd
import json
import os

def extract_id_from_json_str(json_str):
    """
    从 JSON 字符串中解析出 id 字段，并转为字符串格式
    """
    if not isinstance(json_str, str):
        return None
    try:
        data = json.loads(json_str)
        # 确保 id 转为字符串，防止一个是数字一个是字符串导致匹配失败
        return str(data.get('id', ''))
    except:
        return None

def load_forum_data_to_dict(file_paths):
    """
    读取多个论坛文件，返回一个字典：{ 'id_string': 'raw_json_line' }
    """
    id_map = {}
    
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[警告] 找不到文件: {path}，跳过。")
            continue
            
        print(f"正在读取索引库: {path} ...")
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # 提取 id 并转字符串
                    obj_id = str(data.get('id', ''))
                    
                    # 存入字典：id -> 原始行
                    if obj_id:
                        id_map[obj_id] = line
                except json.JSONDecodeError:
                    # 忽略格式错误的行
                    continue
                    
    print(f"索引构建完成，共包含 {len(id_map)} 个唯一 ID。")
    return id_map

def main():
    # --- 文件路径配置 ---
    excel_path = 'matched_raw_output.xlsx'
    
    # 两个论坛文件
    forum_files = [
        'review_forum.jsonl',
        'support_forum.jsonl'
    ]
    
    output_path = 'matched_raw_output_with_forum.xlsx'

    # 1. 读取现有的 Excel
    print(f"正在读取 Excel: {excel_path} ...")
    if not os.path.exists(excel_path):
        print("[错误] 找不到 Excel 文件，请先运行上一步的代码生成它。")
        return
        
    df_excel = pd.read_excel(excel_path)
    print(f"Excel 读取成功，共 {len(df_excel)} 行。")

    # 2. 准备匹配键 (从 Excel 第一列提取 ID)
    # 我们假设第一列是 Feedback 的内容。使用 iloc[:, 0] 选取第一列，不依赖列名。
    print("正在解析 Excel 中的 ID...")
    # 创建一个临时列 'temp_match_id' 用于匹配
    df_excel['temp_match_id'] = df_excel.iloc[:, 0].apply(extract_id_from_json_str)

    # 3. 构建论坛数据的索引 (ID -> 原始文本)
    forum_dict = load_forum_data_to_dict(forum_files)

    # 4. 进行匹配 (Mapping)
    print("正在匹配论坛数据...")
    # 使用 map 函数，根据 id 查找字典中的内容，填入新列
    df_excel['Forum_Json_Content'] = df_excel['temp_match_id'].map(forum_dict)

    # 5. 填充缺失值 (可选)
    # 如果没匹配到，Excel 里会显示空白 (NaN)。如果你想显示 "Not Found"，可以取消下面注释
    # df_excel['Forum_Json_Content'] = df_excel['Forum_Json_Content'].fillna('Not Found')

    # 6. 清理并保存
    # 删除临时的 id 列
    df_final = df_excel.drop(columns=['temp_match_id'])

    print(f"正在保存到新文件: {output_path} ...")
    df_final.to_excel(output_path, index=False)
    
    # 统计匹配率
    matched_count = df_final['Forum_Json_Content'].notna().sum()
    print(f"--- 处理完成 ---")
    print(f"共 {len(df_final)} 行数据，其中 {matched_count} 行成功匹配到了论坛数据。")
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    main()