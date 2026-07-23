import os
import logging
from typing import List, Tuple
from ingestion.chunker import PolicyChunk

logger = logging.getLogger("retrieval.rerank")


class CrossEncoderReranker:
    """Local Cross-Encoder reranker. Disabled automatically when RERANKER_MODEL=none."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = os.getenv("RERANKER_MODEL", model_name)
        self.model = None

        if self.model_name.lower() == "none":
            logger.info("RERANKER_MODEL=none — reranker disabled. Will use BM25+vector RRF scores only.")
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading Cross-Encoder reranker model: {self.model_name} on CPU...")
            self.model = CrossEncoder(self.model_name, device="cpu")
            logger.info("Cross-Encoder reranker model successfully loaded.")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder model: {e}. Reranker disabled.")
            self.model = None

    def rerank(
        self,
        query: str,
        chunks: List[PolicyChunk],
        top_k: int = 5
    ) -> List[Tuple[PolicyChunk, float]]:
        """
        Reranks a list of candidate chunks. Falls back to top_k by insertion order if model is disabled.
        """
        if not chunks:
            return []

        if self.model is None:
            # Reranker disabled — just return top_k from the already-fused RRF results
            logger.info(f"Reranker disabled: returning top {top_k} chunks from RRF fusion scores.")
            return [(chunk, 1.0) for chunk in chunks[:top_k]]

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        scored_chunks = list(zip(chunks, [float(s) for s in scores]))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        results = scored_chunks[:top_k]
        logger.info(f"Reranked {len(chunks)} candidate chunks; returning top {len(results)} matches.")
        return results
