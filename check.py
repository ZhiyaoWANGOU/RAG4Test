from chromadb.config import Settings
from chromadb import Client

client = Client(Settings(
    anonymized_telemetry=False,
    persist_directory="/Users/zhiyaowang/Documents/GitHub/RAG4Test/chroma"
))

print("现有集合：")
for c in client.list_collections():
    print(" -", c.name)
