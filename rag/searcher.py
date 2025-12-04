import json
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .embedding_provider import BaseEmbeddingProvider

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
except ImportError:  # pragma: no cover - handled via requirements
    TfidfVectorizer = None
    linear_kernel = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - handled via requirements
    BM25Okapi = None


class RAGSearcher:
    """
    Base searcher abstraction. Concrete subclasses are responsible for encoding
    text as needed and returning FAISS/database IDs for downstream fetching.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        alpha: float = 0.5,
        **kwargs,
    ):
        if embedding_provider is None and dimension is None:
            raise ValueError("Either embedding_provider or dimension must be provided.")
        self.embedding_provider = embedding_provider
        self.dimension = dimension or embedding_provider.dimension
        self.alpha = alpha

    def add(self, texts: List[str], db_ids: List[int] = None):
        raise NotImplementedError

    def retrieve(self, queries: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

class FAISSRAGSearcher(RAGSearcher):
    def __init__(self, embedding_provider: BaseEmbeddingProvider, dimension: Optional[int] = None, **kwargs):
        super().__init__(dimension=dimension, embedding_provider=embedding_provider, **kwargs)
        self.index = faiss.IndexFlatIP(self.dimension)
        # Mapping from FAISS index position to database ID
        self.faiss_to_db_id: List[int] = []

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_provider.encode(texts, show_progress=False)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def add(self, texts: List[str], db_ids: List[int] = None):
        """
        Add documents to FAISS index after encoding.
        :param texts: raw text chunks
        :param db_ids: list of database IDs corresponding to each text
        """
        if not texts:
            return

        if db_ids is None:
            # If no db_ids provided, use sequential IDs starting from current index size
            start_id = len(self.faiss_to_db_id)
            db_ids = list(range(start_id, start_id + len(texts)))

        embeddings = self._encode(texts)
        self.index.add(embeddings)
        self.faiss_to_db_id.extend(db_ids)

    def retrieve(self, queries: List[str], top_k: int):
        """
        Retrieve top-k similar embeddings and return mapped database IDs.
        :param queries: list of raw query texts
        :param top_k: number of results to return per query
        """
        if len(self.faiss_to_db_id) == 0:
            distances = np.full((len(queries), top_k), np.inf, dtype=np.float32)
            db_indices = np.full((len(queries), top_k), -1, dtype=np.int64)
            return distances, db_indices

        query_embeddings = self._encode(queries)
        scores, faiss_indices = self.index.search(query_embeddings, top_k)
        distances = -scores  # convert similarity to distance-like values
        
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
        
        return distances.astype(np.float32), db_indices.astype(np.int64)

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
        print(f"Here is {path}")
        if os.path.isdir(path) or not path.endswith('.index'):
            os.makedirs(path, exist_ok=True)
            index_file = os.path.join(path, "faiss.index")
            mapping_file = os.path.join(path, "faiss_to_db_id.json")
        else:
            index_file = path
            mapping_file = os.path.join(os.path.dirname(path), "faiss_to_db_id.json")

        print(f"Saving FAISS index to {index_file} ...")
        faiss.write_index(self.index, index_file)

        # Save mapping
        with open(mapping_file, 'w') as f:
            json.dump(self.faiss_to_db_id, f)


class OpenSearchProvider(RAGSearcher):
    def __init__(self, embedding_provider: BaseEmbeddingProvider, dimension: Optional[int] = None,
                 endpoint: str = None, index_name: str = "rag_index",
                 region_name: str = "us-east-1", **kwargs):
        super().__init__(dimension=dimension, embedding_provider=embedding_provider, **kwargs)
        self.endpoint = endpoint
        self.index_name = index_name
        self.region_name = region_name
        
        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection
            from requests_aws4auth import AWS4Auth
            import boto3
            
            # Create AWS auth
            credentials = boto3.Session().get_credentials()
            awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, 
                             region_name, 'es', session_token=credentials.token)
            
            # Create OpenSearch client
            self.client = OpenSearch(
                hosts=[{'host': endpoint.replace('https://', '').replace('http://', ''), 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
        except ImportError:
            raise ImportError(
                "opensearch-py and requests-aws4auth are required for OpenSearch provider. "
                "Install with: pip install opensearch-py requests-aws4auth"
            )
        
        # Initialize index if it doesn't exist
        self._ensure_index()
    
    def _ensure_index(self):
        """Create OpenSearch index if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            # Create index with k-NN mapping
            index_body = {
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.dimension
                        },
                        "db_id": {
                            "type": "integer"
                        }
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=index_body)
    
    def add(self, texts: List[str], db_ids: List[int] = None):
        if db_ids is None:
            try:
                count_response = self.client.count(index=self.index_name)
                start_id = count_response['count']
            except:
                start_id = 0
            db_ids = list(range(start_id, start_id + len(texts)))

        embeddings = self.embedding_provider.encode(texts, show_progress=False)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        embeddings = embeddings.astype(np.float32)
        
        from opensearchpy.helpers import bulk
        actions = []
        for i, (embedding, db_id) in enumerate(zip(embeddings, db_ids)):
            action = {
                "_index": self.index_name,
                "_id": i + len(actions),  # Use sequential OpenSearch IDs
                "_source": {
                    "embedding": embedding.tolist(),
                    "db_id": int(db_id)
                }
            }
            actions.append(action)
        
        if actions:
            bulk(self.client, actions)
            self.client.indices.refresh(index=self.index_name)
    
    def retrieve(self, queries: List[str], top_k: int):
        all_distances = []
        all_db_indices = []
        
        embeddings = self.embedding_provider.encode(queries, show_progress=False)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        embeddings = embeddings.astype(np.float32)
        
        for query_vec in embeddings:
            # k-NN search query
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_vec.tolist(),
                            "k": top_k
                        }
                    }
                },
                "_source": ["db_id"]
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            
            # Extract results
            distances = []
            db_indices = []
            for hit in response['hits']['hits']:
                # OpenSearch k-NN doesn't return distances directly in all versions
                # use score as distance approximation (lower is better for L2)
                score = hit.get('_score', 0.0)
                distances.append(score)
                db_indices.append(hit['_source']['db_id'])
            
            # pad if not enough results
            while len(distances) < top_k:
                distances.append(float('inf'))
                db_indices.append(-1)
            
            all_distances.append(distances[:top_k])
            all_db_indices.append(db_indices[:top_k])
        
        return np.array(all_distances), np.array(all_db_indices)
    
    def load(self, path: str):
        if os.path.isdir(path):
            index_name_file = os.path.join(path, "opensearch_index_name.json")
        else:
            index_name_file = os.path.join(os.path.dirname(path), "opensearch_index_name.json")
        
        # load index name from file if it exists
        if os.path.exists(index_name_file):
            with open(index_name_file, 'r') as f:
                saved_data = json.load(f)
                self.index_name = saved_data.get('index_name', self.index_name)
            print(f"Loaded OpenSearch index name '{self.index_name}' from {index_name_file}")
        else:
            # use UUID to ensure uniqueness across different databases
            unique_id = str(uuid.uuid4()).replace('-', '_')
            self.index_name = f"rag_index_{unique_id}"
            print(f"Generated new OpenSearch index name '{self.index_name}' for path: {path}")
        
        # ensure the index exists with the loaded/generated name
        self._ensure_index()
        print(f"OpenSearch index '{self.index_name}' is ready")
    
    def save(self, path: str):
        if os.path.isdir(path) or not path.endswith('.json'):
            os.makedirs(path, exist_ok=True)
            index_name_file = os.path.join(path, "opensearch_index_name.json")
        else:
            index_name_file = os.path.join(os.path.dirname(path), "opensearch_index_name.json")
        
        # save the current index name to file
        with open(index_name_file, 'w') as f:
            json.dump({'index_name': self.index_name}, f)
        
        # OpenSearch index is automatically persisted, just refresh
        self.client.indices.refresh(index=self.index_name)
        print(f"OpenSearch index '{self.index_name}' saved to {index_name_file} and refreshed")


class HybridRAGSearcher(RAGSearcher):
    """
    Combines sparse (TF-IDF or BM25) and dense FAISS search with weighted fusion.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        dimension: Optional[int] = None,
        sparse_type: str = "tfidf",
        alpha: float = 0.5,
        min_candidate_pool: int = 50,
        **kwargs,
    ):
        super().__init__(dimension=dimension, embedding_provider=embedding_provider, alpha=alpha, **kwargs)
        self.sparse_type = sparse_type.lower()
        if self.sparse_type not in {"tfidf", "bm25"}:
            raise ValueError("sparse_type must be either 'tfidf' or 'bm25'.")
        if self.sparse_type == "tfidf" and (TfidfVectorizer is None or linear_kernel is None):
            raise ImportError("scikit-learn is required for TF-IDF sparse search.")
        if self.sparse_type == "bm25" and BM25Okapi is None:
            raise ImportError("rank-bm25 is required for BM25 sparse search.")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.faiss_to_db_id: List[int] = []
        self.documents: List[str] = []
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._bm25 = None
        self._tokenized_docs: List[List[str]] = []
        self.min_candidate_pool = min_candidate_pool

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_provider.encode(texts, show_progress=False)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_sparse_index(self):
        if self.sparse_type == "tfidf":
            self._tfidf_vectorizer = TfidfVectorizer(stop_words="english")
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(self.documents) if self.documents else None
        else:
            self._tokenized_docs = [self._tokenize(doc) for doc in self.documents]
            self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None

    def add(self, texts: List[str], db_ids: List[int] = None):
        if not texts:
            return

        if db_ids is None:
            start_id = len(self.faiss_to_db_id)
            db_ids = list(range(start_id, start_id + len(texts)))

        embeddings = self._encode(texts)
        self.index.add(embeddings)
        self.faiss_to_db_id.extend(db_ids)
        self.documents.extend(texts)
        self._rebuild_sparse_index()

    def _dense_search(self, query_embeddings: np.ndarray, candidate_k: int):
        scores, indices = self.index.search(query_embeddings, candidate_k)
        # convert similarity to [0,1] via min-max per query after shifting to positive range
        sims = (scores + 1.0) / 2.0  # since cosine similarity in [-1,1]
        return sims, indices

    def _sparse_scores(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.zeros(0, dtype=np.float32)

        if self.sparse_type == "tfidf":
            if self._tfidf_vectorizer is None or self._tfidf_matrix is None:
                return np.zeros(len(self.documents), dtype=np.float32)
            query_vec = self._tfidf_vectorizer.transform([query])
            scores = linear_kernel(query_vec, self._tfidf_matrix).flatten()
            return scores.astype(np.float32)
        else:
            if self._bm25 is None:
                return np.zeros(len(self.documents), dtype=np.float32)
            tokens = self._tokenize(query)
            scores = np.array(self._bm25.get_scores(tokens), dtype=np.float32)
            # print(scores.shape)
            return scores

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        max_val = scores.max()
        min_val = scores.min()
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def retrieve(self, queries: List[str], top_k: int):
        if len(self.faiss_to_db_id) == 0:
            distances = np.full((len(queries), top_k), np.inf, dtype=np.float32)
            db_indices = np.full((len(queries), top_k), -1, dtype=np.int64)
            return distances, db_indices

        candidate_k = min(len(self.faiss_to_db_id), max(self.min_candidate_pool, top_k * 3))
        query_embeddings = self._encode(queries)
        dense_scores, dense_indices = self._dense_search(query_embeddings, candidate_k)

        all_distances = []
        all_db_indices = []

        total_docs = len(self.faiss_to_db_id)

        for i, query in enumerate(queries):
            dense_score_vec = np.zeros(total_docs, dtype=np.float32)
            dense_positions = dense_indices[i]
            valid_mask = dense_positions >= 0
            valid_positions = dense_positions[valid_mask]
            valid_scores = dense_scores[i][valid_mask]
            dense_score_vec[valid_positions] = valid_scores

            sparse_score_vec = self._sparse_scores(query)
            if sparse_score_vec.shape[0] != total_docs:
                sparse_score_vec = np.zeros(total_docs, dtype=np.float32)

            dense_norm = self._normalize(dense_score_vec)
            sparse_norm = self._normalize(sparse_score_vec)
            fused = (1.0 - self.alpha) * dense_norm + self.alpha * sparse_norm

            if np.all(fused == 0):
                # fall back to dense ordering if sparse contributes nothing
                top_positions = valid_positions[:top_k]
                fused_scores = dense_norm[top_positions]
            else:
                top_count = min(top_k, total_docs)
                top_positions = np.argpartition(-fused, top_count - 1)[:top_count]
                top_positions = top_positions[np.argsort(-fused[top_positions])]
                fused_scores = fused[top_positions]

            # pad results if needed
            if len(top_positions) < top_k:
                pad = top_k - len(top_positions)
                top_positions = np.concatenate([top_positions, np.full(pad, -1, dtype=np.int64)])
                fused_scores = np.concatenate([fused_scores, np.zeros(pad, dtype=np.float32)])

            db_ids = [self.faiss_to_db_id[pos] if pos >= 0 else -1 for pos in top_positions[:top_k]]
            distances = 1.0 - fused_scores[:top_k]
            all_db_indices.append(db_ids)
            all_distances.append(distances.tolist())

        return np.array(all_distances, dtype=np.float32), np.array(all_db_indices, dtype=np.int64)

    def load(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "faiss.index")
            mapping_file = os.path.join(path, "faiss_to_db_id.json")
            docs_file = os.path.join(path, "docs.json")
            meta_file = os.path.join(path, "hybrid_meta.json")
        else:
            index_file = path
            base_dir = os.path.dirname(path)
            mapping_file = os.path.join(base_dir, "faiss_to_db_id.json")
            docs_file = os.path.join(base_dir, "docs.json")
            meta_file = os.path.join(base_dir, "hybrid_meta.json")

        self.index = faiss.read_index(index_file)
        with open(mapping_file, "r") as f:
            self.faiss_to_db_id = json.load(f)
        with open(docs_file, "r") as f:
            self.documents = json.load(f)
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                saved = json.load(f)
                self.sparse_type = saved.get("sparse_type", self.sparse_type)
                # self.alpha = saved.get("alpha", self.alpha)

        self._rebuild_sparse_index()

    def save(self, path: str):
        if os.path.isdir(path) or not path.endswith(".index"):
            os.makedirs(path, exist_ok=True)
            index_file = os.path.join(path, "faiss.index")
            mapping_file = os.path.join(path, "faiss_to_db_id.json")
            docs_file = os.path.join(path, "docs.json")
            meta_file = os.path.join(path, "hybrid_meta.json")
        else:
            index_file = path
            base_dir = os.path.dirname(path)
            mapping_file = os.path.join(base_dir, "faiss_to_db_id.json")
            docs_file = os.path.join(base_dir, "docs.json")
            meta_file = os.path.join(base_dir, "hybrid_meta.json")

        faiss.write_index(self.index, index_file)
        with open(mapping_file, "w") as f:
            json.dump(self.faiss_to_db_id, f)
        with open(docs_file, "w") as f:
            json.dump(self.documents, f)
        with open(meta_file, "w") as f:
            json.dump({"sparse_type": self.sparse_type}, f)