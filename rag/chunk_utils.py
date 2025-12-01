import re
import math
from typing import Optional, List

import numpy as np
from sentence_transformers import SentenceTransformer


class ChunkManager:
    def __init__(
        self,
        mode: str,
        embedder: Optional[SentenceTransformer] = None,
        semantic_threshold: float = 0.4,
    ):
        """
        :param mode: "base", "overlap", or "semantic"
        :param embedder: SentenceTransformer model for semantic mode
        :param semantic_threshold: min cosine similarity to keep sentences in same chunk
        """
        self.mode = mode
        self.embedder = embedder
        self.semantic_threshold = semantic_threshold

        if self.mode == "semantic" and self.embedder is None:
            raise ValueError("semantic mode requires a SentenceTransformer embedder.")

    # Splitting sentences
    def split_into_sentences(self, text: str):
        if self.mode in ("base", "overlap", "semantic"):
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            return [s for s in sentences if s]
        else:
            raise ValueError(f"Unknown chunk mode: {self.mode}")

    def _approx_token_len(self, text: str) -> int:
        return len(text.split())

    def _cosine_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def chunk_sentences(
        self,
        text: str,
        max_sentences: int = 5,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ):
        """
        base: non-overlapping 5-sentence blocks (original behavior)

        overlap: sentence-aware chunking with token-based window + overlap

        semantic: sentence-aware chunking with:
            semantic boundaries (cosine similarity threshold)
            token budget
            optional overlap (semantic + overlap)
        """
        sentences = self.split_into_sentences(text)

        if self.mode == "base":
            chunks = []
            for i in range(0, len(sentences), max_sentences):
                chunk = " ".join(sentences[i:i + max_sentences])
                chunks.append(chunk)
            return chunks

        if self.mode == "overlap":
            chunks = []
            current_sents: List[str] = []
            current_tokens = 0

            for sent in sentences:
                sent_tokens = self._approx_token_len(sent)

                if current_sents and current_tokens + sent_tokens > max_tokens:
                    chunks.append(" ".join(current_sents))

                    # create overlap window
                    if overlap_tokens > 0:
                        overlap_sents: List[str] = []
                        token_sum = 0
                        for prev in reversed(current_sents):
                            t = self._approx_token_len(prev)
                            if token_sum + t > overlap_tokens:
                                break
                            overlap_sents.insert(0, prev)  # prepend
                            token_sum += t
                        current_sents = overlap_sents + [sent]
                        current_tokens = sum(self._approx_token_len(s) for s in current_sents)
                    else:
                        current_sents = [sent]
                        current_tokens = sent_tokens
                else:
                    current_sents.append(sent)
                    current_tokens += sent_tokens

            if current_sents:
                chunks.append(" ".join(current_sents))

            return chunks

        if self.mode == "semantic":
            chunks: List[str] = []
            if not sentences:
                return chunks

            # shape: (num_sentences, dim)
            sent_embs = self.embedder.encode(
                sentences,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            current_sents: List[str] = []
            current_idx: List[int] = []   # indices of sentences in current chunk
            current_tokens = 0
            current_emb: Optional[np.ndarray] = None

            for i, sent in enumerate(sentences):
                sent_tokens = self._approx_token_len(sent)
                sent_emb = sent_embs[i]

                if not current_sents:
                    current_sents = [sent]
                    current_idx = [i]
                    current_tokens = sent_tokens
                    current_emb = sent_emb
                    continue

                # compute similarity with current chunk embedding
                sim = self._cosine_sim(sent_emb, current_emb)

                # decide whether to start a new chunk
                start_new = False

                # semantic boundary
                if sim < self.semantic_threshold:
                    start_new = True

                # token budget exceeded
                if current_tokens + sent_tokens > max_tokens:
                    start_new = True

                if start_new:
                    chunks.append(" ".join(current_sents))

                    if overlap_tokens > 0:
                        # build overlap window (sentences + indices)
                        overlap_sents: List[str] = []
                        overlap_idx: List[int] = []
                        token_sum = 0

                        for s, idx_ in reversed(list(zip(current_sents, current_idx))):
                            t = self._approx_token_len(s)
                            if token_sum + t > overlap_tokens:
                                break
                            overlap_sents.insert(0, s)     # prepend to keep order
                            overlap_idx.insert(0, idx_)
                            token_sum += t

                        current_sents = overlap_sents + [sent]
                        current_idx = overlap_idx + [i]
                        current_tokens = sum(self._approx_token_len(s) for s in current_sents)

                        # recompute centroid embedding from sentence indices
                        current_emb = sent_embs[current_idx].mean(axis=0)
                    else:
                        current_sents = [sent]
                        current_idx = [i]
                        current_tokens = sent_tokens
                        current_emb = sent_emb
                else:
                    current_sents.append(sent)
                    current_idx.append(i)
                    current_tokens += sent_tokens
                    current_emb = sent_embs[current_idx].mean(axis=0)

            if current_sents:
                chunks.append(" ".join(current_sents))

            return chunks

        raise ValueError(f"Unknown mode: {self.mode}")
