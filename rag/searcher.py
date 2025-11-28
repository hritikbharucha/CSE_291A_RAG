import faiss
import numpy as np
import os
import json
from typing import List, Tuple, Dict

class RAGSearcher:
    def __init__(self, dimension, **kwargs):
        self.dimension = dimension

    def add(self, embeddings: np.ndarray, db_ids: List[int] = None):
        pass

    def retrieve(self, queries: np.ndarray, top_k: int)-> Tuple[np.ndarray, np.ndarray]:
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
        # Mapping from FAISS index position to database ID
        # faiss_position -> database_id
        self.faiss_to_db_id = []

    def add(self, embeddings: np.ndarray, db_ids: List[int] = None):
        """
        Add embeddings to FAISS index and maintain mapping to database IDs.
        
        Args:
            embeddings: numpy array of embeddings to add
            db_ids: list of database IDs corresponding to each embedding
        """
        if db_ids is None:
            # If no db_ids provided, use sequential IDs starting from current index size
            start_id = len(self.faiss_to_db_id)
            db_ids = list(range(start_id, start_id + len(embeddings)))
        
        self.index.add(embeddings)
        self.faiss_to_db_id.extend(db_ids)

    def retrieve(self, queries: np.ndarray, top_k: int):
        """
        Retrieve top-k similar embeddings and return mapped database IDs.
        
        Args:
            queries: numpy array of query embeddings (2D array: [num_queries, dimension])
            top_k: number of results to return per query
        
        Returns:
            distances: numpy array of distances
            db_indices: numpy array of database IDs (mapped from FAISS indices)
        """
        distances, faiss_indices = self.index.search(queries, top_k)
        
        # Map FAISS indices to database IDs
        if len(self.faiss_to_db_id) > 0:
            # Handle both 1D and 2D arrays
            if faiss_indices.ndim == 1:
                db_indices = np.array([self.faiss_to_db_id[idx] if idx < len(self.faiss_to_db_id) else -1 
                                      for idx in faiss_indices])
            else:
                db_indices = np.array([[self.faiss_to_db_id[idx] if idx < len(self.faiss_to_db_id) else -1 
                                       for idx in row] for row in faiss_indices])
        else:
            # Fallback: if no mapping exists, return FAISS indices as-is
            db_indices = faiss_indices
        
        return distances, db_indices

    def load(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "faiss.index")
            mapping_file = os.path.join(path, "faiss_to_db_id.json")
        else:
            index_file = path
            mapping_file = os.path.join(os.path.dirname(path), "faiss_to_db_id.json")
        
        self.index = faiss.read_index(index_file)
        
        # Load mapping if it exists
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.faiss_to_db_id = json.load(f)
        else:
            # If mapping doesn't exist, assume sequential mapping (for backward compatibility)
            # This might not be correct if database IDs are not sequential
            self.faiss_to_db_id = list(range(self.index.ntotal))
            print(f"Warning: No mapping file found at {mapping_file}. Assuming sequential mapping.")

    def save(self, path: str):
        if os.path.isdir(path) or not path.endswith('.index'):
            os.makedirs(path, exist_ok=True)
            index_file = os.path.join(path, "faiss.index")
            mapping_file = os.path.join(path, "faiss_to_db_id.json")
        else:
            index_file = path
            mapping_file = os.path.join(os.path.dirname(path), "faiss_to_db_id.json")
        
        faiss.write_index(self.index, index_file)
        
        # Save mapping
        with open(mapping_file, 'w') as f:
            json.dump(self.faiss_to_db_id, f)