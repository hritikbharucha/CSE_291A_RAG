import os
import faiss
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer


class RAG:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", dimension: int = 384):
        self.model = SentenceTransformer(embedding_model)
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.docs = []  
        self.doc_metadata = [] 

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        if metadata is None:
            metadata = [{} for _ in documents]

        embeddings = self.model.encode(documents, convert_to_numpy=True)
        self.index.add(embeddings)
        self.docs.extend(documents)
        self.doc_metadata.extend(metadata)
        print(f"Added {len(documents)} documents to FAISS index (total = {len(self.docs)}).")

    def retrieve(self, query: str, top_k_most_similar: int = 5) -> List[Tuple[str, Dict, float]]:
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k_most_similar)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.docs):
                results.append((self.docs[idx], self.doc_metadata[idx], float(dist)))

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        np.save(os.path.join(path, "docs.npy"), np.array(self.docs, dtype=object))
        np.save(os.path.join(path, "meta.npy"), np.array(self.doc_metadata, dtype=object))
        print(f"Index and documents saved to {path}.")

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        self.docs = list(np.load(os.path.join(path, "docs.npy"), allow_pickle=True))
        self.doc_metadata = list(np.load(os.path.join(path, "meta.npy"), allow_pickle=True))
        print(f"Loaded {len(self.docs)} documents from {path}.")


if __name__ == "__main__":
    rag = RAG()

    # Example mock data for testing
    historical_docs = [
        "The 2008 financial crisis was a global economic downturn.",
        "The Great Depression occurred in the 1930s.",
        "The 1973 oil crisis caused inflation worldwide."
    ]
    meta = [{"title": "2008 Crisis"}, {"title": "Great Depression"}, {"title": "1973 Oil Crisis"}]

    rag.add_documents(historical_docs, meta)

    query = "Events about economy"
    results = rag.retrieve(query, top_k_most_similar=2)

    print("\nRetrieval Results:")
    for text, m, score in results:
        print(f" - {m.get('title', 'unknown')} (score={score:.4f})\n   {text[:80]}...")
