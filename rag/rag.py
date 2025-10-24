import torch
import torch.nn as nn
import os
from .lru_cache import LRUCache
from .searcher import *
from .database import *


class RAG:
    def __init__(self,
                 embedding_model: nn.Module, rag_searcher: RAGSearcher,
                 cache_size: int, db_dir: str = None, db_name: str = None, new_db = False,
    **kwargs):
        self.database = None
        self.embedding_model = embedding_model
        self.rag_searcher = rag_searcher
        self.lru_cache = LRUCache(capacity=cache_size)
        self.cache_size = cache_size

        if db_dir is not None and (not new_db):
            self.load(db_dir, db_name)
        else:
            self.database = DocStore()
            self.database.init_schema()
            self.load_from_db()
            self.db_name = "docs"
            self.db_dir = "data"

    def load(self, load_dir: str, db_name: str=None):
        assert os.path.exists(load_dir)
        assert os.path.isdir(load_dir)
        assert os.path.exists(os.path.join(load_dir, "index"))
        assert os.path.exists(os.path.join(load_dir, f"{db_name}.sqlite"))

        print("loading database")
        self.db_dir = load_dir
        self.database = DocStore(os.path.join(load_dir, f"{db_name}.sqlite"))
        self.db_name = "docs"
        self.rag_searcher.load(path=os.path.join(load_dir, "index"))
        print("database loaded")

    def load_from_db(self):
        rows = self.database.top_hot(limit=self.cache_size)
        for row in rows:
            id = row["id"]
            doc = row["doc"]
            meta = row["meta"]
            self.lru_cache.put(id, {"doc": doc, "meta": meta})

    def save(self):
        self.rag_searcher.save(path=os.path.join(self.db_dir, "index"))
        # database should be updated everytime we use

    def add(self, documents: List[str], metadata: List[Dict]=None):
        if not documents:  # Handle empty list
            return
            
        if metadata is None:
            metadata = [{} for _ in documents]

        embeddings = self.embedding_model.encode(documents)
        if type(embeddings) is torch.Tensor:
            embeddings = embeddings.detach().cpu().numpy()

        self.rag_searcher.add(embeddings)

        rows = [{"doc": doc, "meta": meta} for doc, meta in zip(documents, metadata)]
        db_ids = self.database.insert_docs(rows)
        
        # Fix: Handle dict return type properly
        if isinstance(db_ids, dict):
            db_id_list = list(db_ids.keys())
        else:
            db_id_list = db_ids
        
        # Fix: Add bounds checking
        idx = 0
        while len(self.lru_cache._od) < self.cache_size and idx < len(db_id_list):
            self.lru_cache.put(db_id_list[idx], {"doc": documents[idx], "meta": metadata[idx]})
            idx += 1

        print(f"Added {len(documents)} documents.")

    def retrieve(self, queries: List[str], top_k: int) -> Dict[int, Dict[str, Any]]:
        queries = self.embedding_model.encode(queries)
        distances, indices = self.rag_searcher.retrieve(queries, top_k)
        # Convert numpy array to list of integers for hashability
        indices_list = indices.flatten().tolist() if isinstance(indices, np.ndarray) else indices
        cache_rslts, misses = self.lru_cache.get_many(indices_list)
        db_rslts = {}
        if len(misses) > 0:
            db_rslts = self.database.fetch_docs(misses)
            self.lru_cache.put_many(db_rslts)
        return {**db_rslts, **cache_rslts}





