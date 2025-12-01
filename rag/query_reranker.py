from typing import List, Tuple, Optional
import numpy as np
import torch.cuda
from sentence_transformers import CrossEncoder


class Reranker:
    """
    - "BAAI/bge-reranker-v2-m3"
    - "cross-encoder/ms-marco-MiniLM-L12-v2"
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        default_device = "mps" if torch.mps.is_available() else default_device
        device = device if device is not None else default_device
        self.model = CrossEncoder(model_name, device=device)

    def score(
        self,
        query: str,
        docs: List[str],
    ) -> List[float]:
        if not docs:
            return []

        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in np.asarray(scores).tolist()]

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
