import faiss
import numpy as np
import os
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
        if os.path.isdir(path):
            index_file = os.path.join(path, "faiss.index")
        else:
            index_file = path
        self.index = faiss.read_index(index_file)

    def save(self, path: str):
        if os.path.isdir(path) or not path.endswith('.index'):
            os.makedirs(path, exist_ok=True)
            index_file = os.path.join(path, "faiss.index")
        else:
            index_file = path
        faiss.write_index(self.index, index_file)