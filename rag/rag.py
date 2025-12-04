import os
import numpy as np
from tqdm import tqdm
from .lru_cache import LRUCache
from .searcher import *
from .database import *
from rag.aws_config import get_bedrock_llm_model, get_aws_region, list_available_bedrock_models
import warnings


class RAG:
    def __init__(self,
                 embedding_model=None, rag_searcher: RAGSearcher = None,
                 cache_size: int = 0, db_dir: str = "data", db_name: str = "docs", new_db = False,
                #  mode="query_refiner+reranker",
                 mode="base",
                #  mode="query_refiner",
                #  mode="reranker",
                 add_batch_size=32,
                 **kwargs):
        if rag_searcher is None:
            raise ValueError("rag_searcher must be provided.")
        if embedding_model is not None:
            warnings.warn(
                "embedding_model is no longer used directly by RAG. "
                "Pass embedding providers to the searcher instead.",
                DeprecationWarning
            )

        self.database = None
        self.rag_searcher = rag_searcher
        self.lru_cache = LRUCache(capacity=cache_size)
        self.cache_size = cache_size
        self.db_dir = db_dir
        self.db_name = db_name
        self.mode = mode
        self.load(db_dir, db_name)
        self.add_batch_size = add_batch_size

        self.query_refiner = None
        self.reranker = None
        if 'query_refiner' in self.mode:
            from . import query_refiner
            from .llm_provider import create_llm_provider
            # deepseek-ai/deepseek-r1-distill-qwen-1.5b Not a good idea to think
            # microsoft/Phi-3-mini-128k-instruct
            # QWen2.5-3B
            llm = create_llm_provider(provider_type="huggingface", model_name="microsoft/Phi-3-mini-128k-instruct")

            # AWS
            # model_name = get_bedrock_llm_model("mistral-7b")
            # llm = create_llm_provider(provider_type="bedrock", model_name=model_name, region_name=get_aws_region())

            self.query_refiner = query_refiner.QueryRefiner(llm_provider=llm)
        if 'reranker' in self.mode:
            # print(list_available_bedrock_models())
            from . import query_reranker
            # self.reranker = query_reranker.Reranker(region=get_aws_region())
            # self.reranker = query_reranker.BedrockReranker(region=get_aws_region())
            self.reranker = query_reranker.HFReranker()

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
        # If cache is disabled, skip any preloading work
        if self.cache_size <= 0:
            return

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

        print(self.rag_searcher)
        print(f"Indexing {len(chunks)} chunks with searcher...")
        for st in tqdm.trange(0, len(chunks), self.add_batch_size):
            batch_chunks = chunks[st:st+self.add_batch_size]
            batch_ids = indices[st:st+self.add_batch_size]
            self.rag_searcher.add(batch_chunks, db_ids=batch_ids)

    def retrieve(self, queries: List[str], top_k: int) -> Dict[int, Dict[str, Any]]:
        if self.query_refiner is not None:
            queries = [self.query_refiner.get_refined_query(query) for query in queries]

        db_topk = top_k if self.reranker is None else 30
        distances, indices = self.rag_searcher.retrieve(queries, db_topk)

        # Convert numpy array to list of integers for hashability
        indices_list = indices.flatten().tolist() if isinstance(indices, np.ndarray) else indices

        # when self.cache_size <= 0, we disable cache
        if self.cache_size <= 0:
            cache_rslts = {}
            db_rslts = self.database.fetch_chunks(indices_list)
        else:
            cache_rslts, misses = self.lru_cache.get_many(indices_list)
            db_rslts = {}
            if len(misses) > 0:
                db_rslts = self.database.fetch_chunks(misses)
                self.lru_cache.put_many(db_rslts)

        if self.reranker is not None:
            # in our case, len(queries) is always 1
            docs = [db_rslts[key]["doc"] for key in db_rslts.keys()]
            ranked = self.reranker.rerank(queries[0], docs, top_k)
            keys = list(db_rslts.keys())
            # print(ranked)
            # Convert ranked results back to dict format
            db_rslts = {keys[ranked_item[0]]: db_rslts[keys[ranked_item[0]]] for ranked_item in ranked}
            # Update indices_list to reflect reranked order, limited to top_k
            indices_list = [keys[ranked_item[0]] for ranked_item in ranked]

        indices_list = indices_list[:top_k]

        result = {}
        all_results = {**db_rslts, **cache_rslts}
        for idx in indices_list:
            if idx in all_results:
                result[idx] = all_results[idx]
        
        return result





