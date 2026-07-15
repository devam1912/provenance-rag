import os
import logging
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from ingestion.chunker import PolicyChunk

logger = logging.getLogger("retrieval.rerank")

class CrossEncoderReranker:
    """Local Cross-Encoder reranker using cross-encoder/ms-marco-MiniLM-L-6-v2 for precise document scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Allow loading model name from env if available
        self.model_name = os.getenv("RERANKER_MODEL", model_name)
        logger.info(f"Loading Cross-Encoder reranker model: {self.model_name} on CPU...")
        
        # Load local CrossEncoder model. CPU device is default and safe across machines.
        self.model = CrossEncoder(self.model_name, device="cpu")
        logger.info("Cross-Encoder reranker model successfully loaded.")

    def rerank(
        self,
        query: str,
        chunks: List[PolicyChunk],
        top_k: int = 5
    ) -> List[Tuple[PolicyChunk, float]]:
        """
        Reranks a list of candidate chunks against the user query.
        
        Args:
            query: The user query string.
            chunks: A list of candidate PolicyChunk objects.
            top_k: The number of top results to return.
            
        Returns:
            A list of (PolicyChunk, score) sorted descending by relevance score.
        """
        if not chunks:
            logger.warning("Empty candidate list provided to CrossEncoderReranker. Returning empty list.")
            return []

        # Pairs of (query, document_text) for the cross encoder input
        pairs = [(query, chunk.text) for chunk in chunks]
        
        # Predict relevance scores. Returns a list of floats (logits or sigmoid outputs depending on model).
        scores = self.model.predict(pairs)
        
        # Pair documents with their predicted scores
        scored_chunks = list(zip(chunks, [float(s) for s in scores]))
        
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N results
        results = scored_chunks[:top_k]
        logger.info(f"Reranked {len(chunks)} candidate chunks; returning top {len(results)} matches.")
        return results
