import pytest
from ingestion.chunker import PolicyChunk
from retrieval.fusion import reciprocal_rank_fusion

def create_mock_chunk(chunk_id: str) -> PolicyChunk:
    """Helper to create a basic mock PolicyChunk."""
    return PolicyChunk(
        chunk_id=chunk_id,
        text=f"Sample text for {chunk_id}",
        source="test_policy.txt",
        headers=["Test Header"],
        chunk_index=0
    )

def test_rrf_empty_inputs():
    """RRF should return an empty list when both input lists are empty."""
    results = reciprocal_rank_fusion([], [])
    assert results == []

def test_rrf_symmetric_ranks():
    """
    Symmetric ranks (e.g. Doc A at rank 1/2 and Doc B at rank 2/1)
    should receive the same RRF score and sort alphabetically by chunk_id.
    """
    chunk_a = create_mock_chunk("chunk_a")
    chunk_b = create_mock_chunk("chunk_b")

    # chunk_a: rank 1 sparse, rank 2 dense
    # chunk_b: rank 2 sparse, rank 1 dense
    sparse = [(chunk_a, 10.0), (chunk_b, 5.0)]
    dense = [(chunk_b, 0.9), (chunk_a, 0.1)]

    results = reciprocal_rank_fusion(sparse, dense, k=60)
    
    assert len(results) == 2
    # Both should have exactly the same score
    score_a = results[0][1]
    score_b = results[1][1]
    assert score_a == score_b
    assert pytest.approx(score_a, abs=1e-6) == (1/61 + 1/62)
    
    # Sort order should fall back to alphabetical (chunk_a first)
    assert results[0][0].chunk_id == "chunk_a"
    assert results[1][0].chunk_id == "chunk_b"

def test_rrf_disjoint_sets():
    """RRF should correctly compute ranks even if the retrieved documents do not overlap."""
    chunk_a = create_mock_chunk("chunk_a")
    chunk_b = create_mock_chunk("chunk_b")

    sparse = [(chunk_a, 10.0)]
    dense = [(chunk_b, 0.9)]

    # chunk_a: rank 1 (1 / (60 + 1))
    # chunk_b: rank 1 (1 / (60 + 1))
    results = reciprocal_rank_fusion(sparse, dense, k=60)
    
    assert len(results) == 2
    assert results[0][0].chunk_id == "chunk_a"  # Alphabetical ordering because scores are tied
    assert results[0][1] == pytest.approx(1/61, abs=1e-6)

def test_rrf_single_channel_empty():
    """RRF should function correctly when one of the retrievers returns zero results."""
    chunk_a = create_mock_chunk("chunk_a")
    chunk_b = create_mock_chunk("chunk_b")

    sparse = [(chunk_a, 10.0), (chunk_b, 5.0)]
    dense = []

    results = reciprocal_rank_fusion(sparse, dense, k=60)
    
    assert len(results) == 2
    assert results[0][0].chunk_id == "chunk_a"
    assert results[0][1] == pytest.approx(1/61, abs=1e-6)
    assert results[1][0].chunk_id == "chunk_b"
    assert results[1][1] == pytest.approx(1/62, abs=1e-6)

def test_rrf_custom_k():
    """Verifies that RRF uses the customized k parameter correctly."""
    chunk_a = create_mock_chunk("chunk_a")
    
    sparse = [(chunk_a, 1.0)]
    dense = [(chunk_a, 1.0)]
    
    # score = 1/(k + 1) + 1/(k + 1) = 2/(k + 1)
    # If k = 10, score = 2 / 11 = 0.181818...
    results = reciprocal_rank_fusion(sparse, dense, k=10)
    
    assert len(results) == 1
    assert results[0][0].chunk_id == "chunk_a"
    assert results[0][1] == pytest.approx(2/11, abs=1e-6)
