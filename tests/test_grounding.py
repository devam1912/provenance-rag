import pytest
from unittest.mock import MagicMock
from ingestion.chunker import PolicyChunk
from grounding.citation_validator import validate_citations, get_cited_sentence

# Mock classes for LLM response
class MockMessage:
    def __init__(self, content: str):
        self.content = content

def test_get_cited_sentence():
    # Test typical sentence boundary extraction
    text = "First sentence. Second sentence with claim [doc.txt#chunk_1]. Third sentence."
    # Index of "[doc.txt#chunk_1]" in text is 42
    match_start = text.find("[doc.txt#chunk_1]")
    sentence = get_cited_sentence(text, match_start)
    assert sentence == "Second sentence with claim"

    # Test starting sentence
    text2 = "Claim at start [doc.txt#chunk_0]."
    match_start2 = text2.find("[doc.txt#chunk_0]")
    sentence2 = get_cited_sentence(text2, match_start2)
    assert sentence2 == "Claim at start"

    # Test newline boundary
    text3 = "Line one\nClaim after newline [doc.txt#chunk_2]."
    match_start3 = text3.find("[doc.txt#chunk_2]")
    sentence3 = get_cited_sentence(text3, match_start3)
    assert sentence3 == "Claim after newline"


def test_validate_citations_success():
    chunk1 = PolicyChunk(
        chunk_id="doc1.txt#chunk_0",
        text="Students must maintain a GPA of 2.0 to avoid academic probation.",
        source="doc1.txt",
        headers=["Standing"],
        chunk_index=0
    )
    chunk2 = PolicyChunk(
        chunk_id="doc1.txt#chunk_1",
        text="The credit transfer limit is 70 semester hours.",
        source="doc1.txt",
        headers=["Transfer"],
        chunk_index=1
    )
    
    retrieved = [chunk1, chunk2]
    answer = "A minimum GPA of 2.0 is required [doc1.txt#chunk_0]. Also, you can transfer up to 70 hours [doc1.txt#chunk_1]."
    
    # Mock LLM to return YES for entailment
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MockMessage("YES")
    
    result = validate_citations(answer, retrieved, mock_llm)
    
    assert result["is_valid"] is True
    assert len(result["failed_citations"]) == 0
    assert mock_llm.invoke.call_count == 2


def test_validate_citations_missing_chunk():
    chunk1 = PolicyChunk(
        chunk_id="doc1.txt#chunk_0",
        text="Students must maintain a GPA of 2.0 to avoid academic probation.",
        source="doc1.txt",
        headers=["Standing"],
        chunk_index=0
    )
    
    retrieved = [chunk1]
    # doc1.txt#chunk_99 was not retrieved
    answer = "Students must keep a 2.0 GPA [doc1.txt#chunk_99]."
    
    mock_llm = MagicMock()
    result = validate_citations(answer, retrieved, mock_llm)
    
    assert result["is_valid"] is False
    assert len(result["failed_citations"]) == 1
    assert "never retrieved" in result["failed_citations"][0]
    # Should not invoke LLM because existence check failed first
    mock_llm.invoke.assert_not_called()


def test_validate_citations_entailment_failure():
    chunk1 = PolicyChunk(
        chunk_id="doc1.txt#chunk_0",
        text="Students must maintain a GPA of 2.0 to avoid academic probation.",
        source="doc1.txt",
        headers=["Standing"],
        chunk_index=0
    )
    
    retrieved = [chunk1]
    # Claim is contradictory/unsupported
    answer = "Students must maintain a 4.0 GPA [doc1.txt#chunk_0]."
    
    # Mock LLM to return NO (or anything else without YES)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MockMessage("NO")
    
    result = validate_citations(answer, retrieved, mock_llm)
    
    assert result["is_valid"] is False
    assert len(result["failed_citations"]) == 1
    assert "not supported" in result["failed_citations"][0]


def test_validate_citations_missing_all_citations():
    chunk1 = PolicyChunk(
        chunk_id="doc1.txt#chunk_0",
        text="Students must maintain a GPA of 2.0 to avoid academic probation.",
        source="doc1.txt",
        headers=["Standing"],
        chunk_index=0
    )
    
    retrieved = [chunk1]
    # Long text making claims but missing citations entirely
    answer = "Academic policies require undergraduate students to maintain a cumulative grade point average of at least 2.0. If their GPA falls below this requirement, they will be placed on academic warning or probation."
    
    mock_llm = MagicMock()
    result = validate_citations(answer, retrieved, mock_llm)
    
    assert result["is_valid"] is False
    assert len(result["failed_citations"]) == 1
    assert "did not include any citations" in result["failed_citations"][0]


def test_validate_citations_fallback_exception():
    # If the response is the fallback "insufficient information", it shouldn't fail validation
    retrieved = []
    answer = "Insufficient verified information is available to answer the query."
    
    mock_llm = MagicMock()
    result = validate_citations(answer, retrieved, mock_llm)
    
    assert result["is_valid"] is True
    assert len(result["failed_citations"]) == 0
