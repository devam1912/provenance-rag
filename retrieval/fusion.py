import logging
from typing import List, Tuple, Dict
from ingestion.chunker import PolicyChunk

logger = logging.getLogger("retrieval.fusion")

def reciprocal_rank_fusion(
    sparse_results: List[Tuple[PolicyChunk, float]],
    dense_results: List[Tuple[PolicyChunk, float]],
    k: int = 60
) -> List[Tuple[PolicyChunk, float]]:
    """
    Applies Reciprocal Rank Fusion (RRF) to merge sparse and dense retrieval results.
    
    Formula: RRF_Score(doc) = Sum_{retriever} (1 / (k + rank_doc))
    where rank_doc is the 1-based rank (position) of the document in the retriever's list.
    
    Args:
        sparse_results: Chunks retrieved by BM25 with their scores, ordered by relevance.
        dense_results: Chunks retrieved by Vector search with their scores, ordered by relevance.
        k: A constant penalizing lower-ranked documents. Defaults to 60.
        
    Returns:
        A list of (PolicyChunk, rrf_score) sorted descending by RRF score.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, PolicyChunk] = {}

    # Helper to calculate rank contributions
    def add_ranks(results: List[Tuple[PolicyChunk, float]]) -> None:
        for rank_idx, (chunk, _) in enumerate(results):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            
            # rank is 1-based index (rank_idx + 1)
            rank = rank_idx + 1
            score_contribution = 1.0 / (k + rank)
            
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_contribution

    # Process sparse and dense results
    add_ranks(sparse_results)
    add_ranks(dense_results)

    # Convert mapping back to list of tuples (PolicyChunk, rrf_score)
    fused_results = [
        (chunk_map[chunk_id], score)
        for chunk_id, score in rrf_scores.items()
    ]

    # Sort descending by score. If scores are equal, sort alphabetically by chunk_id for stability
    fused_results.sort(key=lambda x: (-x[1], x[0].chunk_id))
    
    logger.info(f"Fused {len(sparse_results)} sparse and {len(dense_results)} dense results into {len(fused_results)} unique items.")
    return fused_results
