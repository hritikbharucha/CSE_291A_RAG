from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
from tqdm import tqdm

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        :param texts: list of text strings to encode
        :param show_progress: whether to show progress bar
        :return: numpy array of shape (n_texts, dimension) with embeddings
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass


class SentenceTransformerProvider(BaseEmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, **kwargs)
        self._dimension = self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        if show_progress and len(texts) > 1:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    MODEL_DIMENSIONS = {
        "amazon.titan-embed-text-v1": 1536,
        "amazon.titan-embed-text-v2": 1024,
        "cohere.embed-english-v3": 1024,
        "cohere.embed-multilingual-v3": 1024,
    }
    
    def __init__(self, model_name: str, region_name: str = "us-east-1", **kwargs):
        import boto3
        self.model_name = model_name
        self.region_name = region_name
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1024)
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        import json
        embeddings = []
        iterator = tqdm(texts, desc="Encoding embeddings", disable=not show_progress or len(texts) <= 1) if show_progress and len(texts) > 1 else texts
        
        for text in iterator:
            if self.model_name.startswith("amazon.titan"):
                body = json.dumps({"inputText": text})
            elif self.model_name.startswith("cohere"):
                body = json.dumps({"texts": [text]})
            else:
                raise ValueError(f"Unknown Bedrock model: {self.model_name}")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_name,
                body=body.encode('utf-8')
            )
            
            response_body = json.loads(response['body'].read())
            
            if self.model_name.startswith("amazon.titan"):
                embedding = response_body.get("embedding", [])
            elif self.model_name.startswith("cohere"):
                embedding = response_body.get("embeddings", [])[0]
            else:
                embedding = []
            
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedding_provider(provider_type: str = "sentence_transformer", 
                              model_name: str = None, 
                              **kwargs) -> BaseEmbeddingProvider:
    if provider_type == "sentence_transformer":
        if model_name is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerProvider(model_name, **kwargs)
    elif provider_type == "bedrock":
        if model_name is None:
            model_name = "amazon.titan-embed-text-v1"
        return BedrockEmbeddingProvider(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown embedding provider type: {provider_type}")


def auto_detect_embedding_provider(embedding_model) -> BaseEmbeddingProvider:
    if isinstance(embedding_model, BaseEmbeddingProvider):
        return embedding_model
    
    # for backward compatibility
    try:
        from sentence_transformers import SentenceTransformer
        if isinstance(embedding_model, SentenceTransformer):
            provider = SentenceTransformerProvider.__new__(SentenceTransformerProvider)
            provider.model = embedding_model
            provider._dimension = embedding_model.get_sentence_embedding_dimension()
            return provider
    except ImportError:
        pass
    
    raise TypeError(
        f"embedding_model must be BaseEmbeddingProvider or SentenceTransformer, "
        f"got {type(embedding_model)}"
    )

