import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class Reranker(ABC):
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    @abstractmethod
    def score(self, query: str, docs: List[str]) -> List[float]:
        """
        Compute relevance scores for documents given a query.
        
        Args:
            query: The search query
            docs: List of documents to score
            
        Returns:
            List of relevance scores (one per document)
        """
        pass

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float, str]]:
        """
        Rerank documents by relevance score.
        
        Args:
            query: The search query
            docs: List of documents to rerank
            top_k: Optional limit on number of results to return
            
        Returns:
            List of tuples (original_index, score, document) sorted by score (descending)
        """
        if not docs:
            return []

        scores = self.score(query, docs)
        indices = list(range(len(docs)))

        ranked = sorted(
            zip(indices, scores, docs),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked


class BedrockReranker(Reranker):
    """
    Reranker using Amazon Bedrock model: amazon.rerank-v1:0
    """

    def __init__(
        self,
        model_name: str = "amazon.rerank-v1:0",
        region: str = "us-east-1",
        batch_size: int = 32,
    ):
        super().__init__(batch_size=batch_size)
        self.model_name = model_name

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region
        )

    def _bedrock_rerank_batch(self, query: str, docs: List[str]) -> List[float]:
        payload = {
            "query": query,
            "documents": docs,
        }

        response = self.client.invoke_model(
            modelId=self.model_name,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        body = json.loads(response["body"].read())

        scores = [item["relevance_score"] for item in body["results"]]

        return scores

    def score(self, query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []

        all_scores = []

        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            scores = self._bedrock_rerank_batch(query, batch)
            all_scores.extend(scores)

        return [float(s) for s in all_scores]


class HFReranker(Reranker):
    """
    Local reranker using Hugging Face cross-encoder models.
    Supports models like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - BAAI/bge-reranker-base
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 1,
        device: Optional[str] = None,
    ):
        super().__init__(batch_size=batch_size)
        self.model_name = model_name
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _hf_rerank_batch(self, query: str, docs: List[str]) -> List[float]:
        """
        Score a batch of documents using the Hugging Face model.
        Processes documents in sub-batches of size self.batch_size to manage memory.
        """
        all_scores = []
        
        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i : i + self.batch_size]
            pairs = [[query, doc] for doc in batch_docs]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                
                # Convert logits to scores using sigmoid
                batch_scores = torch.sigmoid(logits).cpu().tolist()
                
                # Ensure we return a list
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores]
                
                all_scores.extend(batch_scores)
        
        return all_scores

    def score(self, query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []

        all_scores = []

        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            scores = self._hf_rerank_batch(query, batch)
            all_scores.extend(scores)

        return [float(s) for s in all_scores]
