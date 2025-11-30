import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from .lru_cache import LRUCache
from .searcher import *
from .database import *
from .embedding_provider import BaseEmbeddingProvider, auto_detect_embedding_provider


class RAG:
    def __init__(self,
                 embedding_model, rag_searcher: RAGSearcher,
                 cache_size: int, db_dir: str = "data", db_name: str = "docs", new_db = False,
    **kwargs):
        self.database = None
        # for backward compatibility that we only use HF model
        self.embedding_model = auto_detect_embedding_provider(embedding_model)
        self.rag_searcher = rag_searcher
        self.lru_cache = LRUCache(capacity=cache_size)
        self.cache_size = cache_size
        self.db_dir = db_dir
        self.db_name = db_name
        self.load(db_dir, db_name)

    def load(self, load_dir: str, db_name: str=None):
        if os.path.exists(load_dir):
            assert os.path.isdir(load_dir)
        else:
            os.makedirs(load_dir)

        print("loading database")
        if os.path.exists(os.path.join(load_dir, f"{db_name}.sqlite")):
            self.db_dir = load_dir
            self.database = DocStore(os.path.join(load_dir, f"{db_name}.sqlite"))
            print("database loaded")
        else:
            self.database = DocStore()
            print("new database")
        # no matter what, init it; if exists, this line do nothing
        self.database.init_schema()

        if os.path.exists(os.path.join(load_dir, "index")):
            self.db_name = "docs"
            self.rag_searcher.load(path=os.path.join(load_dir, "index"))
            print("indexing loaded")



    def load_from_db(self):
        rows = self.database.top_hot(limit=self.cache_size)
        for row in rows:
            id = row["id"]
            doc = row["doc"]
            meta = row["meta"]
            article_id = row["article_id"]
            self.lru_cache.put(id, {"doc": doc, "meta": meta, "article_id": article_id})

    def save(self):
        self.rag_searcher.save(path=os.path.join(self.db_dir, "index"))
        print(f"indexing saved to {os.path.join(self.db_dir, 'index')}")
        # database should be already updated everytime we use

    def add(self, documents: List[str], article_ids: List[str], metadata: List[Dict]=None):
        if not documents:  # Handle empty list
            return
            
        if metadata is None:
            metadata = [{} for _ in documents]

        rows = [{"doc": doc, "meta": meta, "article_id": article_id} for doc, meta, article_id in zip(documents, metadata, article_ids)]
        db_inserted_rows = self.database.insert_docs(rows)
        indices = sorted(db_inserted_rows.keys())
        
        cnt = 0
        while len(self.lru_cache._od) < self.cache_size and cnt < len(db_inserted_rows):
            self.lru_cache.put(
                indices[cnt], {
                    "doc": db_inserted_rows[cnt]['doc'],
                    "meta": db_inserted_rows[cnt]['meta'],
                    "article_id": db_inserted_rows[cnt]['article_id']
                })
            cnt += 1

        print(f"Added {len(documents)} documents .")
        print(f"Added {len(db_inserted_rows)} chunks")

        chunks = [db_inserted_rows[idx]['doc'] for idx in indices]

        print(f"Encoding {len(chunks)} chunks into embeddings...")
        embeddings = self.embedding_model.encode(chunks, show_progress=True)
        if type(embeddings) is torch.Tensor:
            embeddings = embeddings.detach().cpu().numpy()

        print(f"Adding {len(embeddings)} embeddings to index...")
        self.rag_searcher.add(embeddings, db_ids=indices)

    def retrieve(self, queries: List[str], top_k: int) -> Dict[int, Dict[str, Any]]:
        query_embeddings = self.embedding_model.encode(queries, show_progress=False)
        distances, indices = self.rag_searcher.retrieve(query_embeddings, top_k)
        # Convert numpy array to list of integers for hashability
        indices_list = indices.flatten().tolist() if isinstance(indices, np.ndarray) else indices
        cache_rslts, misses = self.lru_cache.get_many(indices_list)
        db_rslts = {}
        if len(misses) > 0:
            db_rslts = self.database.fetch_chunks(misses)
            self.lru_cache.put_many(db_rslts)
        return {**db_rslts, **cache_rslts}





