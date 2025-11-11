from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 初始化 Embedding 模型
embedder = OllamaEmbeddings(model="nomic-embed-text")

# 加载持久化的 Chroma 向量库
store = Chroma(
    collection_name="generated_reports",
    embedding_function=embedder,
    persist_directory="logs/generated_vec"
)

# 查看数据库文档数量
docs = store.get()["documents"]
print("📦 Document count:", len(docs))

# 查询相似内容
query = "The app crashes when you click the settings page."
results = store.similarity_search_with_score(query, k=3)

# 输出相似度结果（注意：score 是距离，要转成 similarity）
if not results:
    print("⚠️ No matching results found.")
else:
    print(f"\n🔍 Query: {query}\n")
    for i, (doc, distance) in enumerate(results):
        similarity = 1 - min(distance, 1)  # 距离转相似度
        print(f"[{i+1}] Similarity={similarity:.3f} | Distance={distance:.3f}")
        print(f"    Feedback: {doc.page_content[:100]}...\n")