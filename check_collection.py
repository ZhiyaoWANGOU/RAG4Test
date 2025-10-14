import chromadb

# 连接到数据库
client = chromadb.PersistentClient(path="./kb_index")

# 列出所有 collection
collections = client.list_collections()
for c in collections:
    print(c.name)
    print(c.name, c.metadata)
