import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RetrieverAgent:
    def __init__(self):
        self.vector_store_path = "faiss_index"
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None

    def _load_index(self):
        if self.vector_store is None:
            if os.path.exists(self.vector_store_path):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # Local file, safe
                )
            else:
                print("Error: Vector store not found. Please run ingest.py first.")

    def retrieve(self, state: dict) -> dict:
        """
        Retrieves relevant documents based on the query.
        """
        print(f"\n--- [Retriever] Searching for context... ---")
        self._load_index()
        
        if not self.vector_store:
            return {"retrieved_chunks": []}

        query = state.get("query", "")
        # Basic Similarity Search
        # Retrieve top 5 chunks to ensure we have enough context
        docs = self.vector_store.similarity_search_with_score(query, k=5)
        
        retrieved_chunks = []
        for doc, score in docs:
            # Score is L2 distance for FAISS (lower is better)
            # We can print it for debugging/observability
            # Filter out very poor matches if needed, but let's keep it simple
            chunk_data = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "score": float(score)
            }
            retrieved_chunks.append(chunk_data)
            print(f"Found chunk (Score: {score:.4f}): {doc.page_content[:50]}...")

        return {"retrieved_chunks": retrieved_chunks}
