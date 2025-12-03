import json
from typing import List, Tuple, Optional
import boto3


class Reranker:
    """
    Reranker using Amazon Bedrock model: amazon.rerank-v1:0
    With detailed debugging prints.
    """

    def __init__(
        self,
        model_name: str = "amazon.rerank-v1:0",
        region: str = "us-east-1",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

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

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float, str]]:
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
