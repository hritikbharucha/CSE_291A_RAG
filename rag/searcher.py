import faiss
import numpy as np
from typing import List, Tuple, Dict

class RAGSearcher:
    def __init__(self, dimension, **kwargs):
        self.dimension = dimension

    def add(self, embeddings: np.ndarray):
        pass

    def retrieve(self, queries: List[np.ndarray], top_k: int)-> Tuple[np.ndarray, np.ndarray]:
        pass

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass

class FAISSRAGSearcher(RAGSearcher):
    def __init__(self, dimension, **kwargs):
        super().__init__(dimension, **kwargs)
        self.index = faiss.IndexFlatL2(dimension)
        self.docs = []
        self.doc_metadata = []

    def add(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def retrieve(self, queries: List[np.ndarray], top_k: int):
        distances, indices = self.index.search(queries, top_k)
        return distances, indices

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def save(self, path: str):
        faiss.write_index(self.index, path)