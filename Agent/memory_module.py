# memory_module.py
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


class SemanticMemory:
    def __init__(self, path="semantic_memory"):
        """Initialize semantic memory with Chroma + Ollama embeddings"""
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")
        self.store = Chroma(
            collection_name="bug_memory",
            embedding_function=self.embedding,
            persist_directory=path
        )
    # add entry to memory
    def add_entry(self, feedback: str, output: str):
        """Store feedback + generated bug report as embeddings"""
        doc_text = f"Feedback: {feedback}\nReport: {output}"
        self.store.add_texts([doc_text])
        self.store.persist()
        print(" Semantic memory updated!")

    def search_similar(self, query: str, top_k: int = 3):
        """Retrieve most semantically similar past cases"""
        results = self.store.similarity_search(query, k=top_k)
        return [r.page_content for r in results]
