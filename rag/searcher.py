import faiss
import numpy as np
import os
import json
import uuid
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
        :param embeddings: numpy array of embeddings to add
        :param db_ids: list of database IDs corresponding to each embedding
        """
        if db_ids is None:
            # If no db_ids provided, use sequential IDs starting from current index size
            start_id = len(self.faiss_to_db_id)
            db_ids = list(range(start_id, start_id + len(embeddings)))

        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.faiss_to_db_id.extend(db_ids)

    def retrieve(self, queries: np.ndarray, top_k: int):
        """
        Retrieve top-k similar embeddings and return mapped database IDs.
        :param queries: numpy array of query embeddings (2D array: [num_queries, dimension])
        :param top_k: number of results to return per query
        :return:
            distances: numpy array of distances
            db_indices: numpy array of database IDs (mapped from FAISS indices)
        """
        faiss.normalize_L2(queries)
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


class OpenSearchProvider(RAGSearcher):
    def __init__(self, dimension, endpoint: str = None, index_name: str = "rag_index", 
                 region_name: str = "us-east-1", **kwargs):
        super().__init__(dimension, **kwargs)
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
    
    def add(self, embeddings: np.ndarray, db_ids: List[int] = None):
        if db_ids is None:
            try:
                count_response = self.client.count(index=self.index_name)
                start_id = count_response['count']
            except:
                start_id = 0
            db_ids = list(range(start_id, start_id + len(embeddings)))
        
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
    
    def retrieve(self, queries: np.ndarray, top_k: int):
        all_distances = []
        all_db_indices = []
        
        for query in queries:
            # k-NN search query
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query.tolist(),
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