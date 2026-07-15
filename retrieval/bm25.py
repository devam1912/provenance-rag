import os
import pickle
import re
import logging
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from ingestion.chunker import PolicyChunk

logger = logging.getLogger("retrieval.bm25")

# Simple list of English stopwords to improve keyword search precision
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cant", "cannot",
    "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few",
    "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell",
    "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill",
    "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so",
    "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there",
    "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent", "what",
    "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with",
    "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
}

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text: lowercases, removes punctuation, and filters stopwords."""
    text = text.lower()
    # Replace non-alphanumeric characters with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]


class BM25Retriever:
    """Sparse retriever wrapper around rank_bm25 for exact keyword matching."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[PolicyChunk] = []

    def build(self, chunks: List[PolicyChunk]) -> None:
        """Builds a BM25 index from a list of PolicyChunks."""
        if not chunks:
            logger.warning("Empty chunk list provided to build BM25 index.")
            return

        self.chunks = chunks
        corpus_tokens = [tokenize_text(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Successfully built BM25 index with {len(chunks)} chunks.")

    def save(self, filepath: str) -> None:
        """Serializes the current index and source chunks to disk."""
        if self.bm25 is None or not self.chunks:
            raise ValueError("Cannot save an unbuilt BM25 index.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # We save the model and the chunks together
        data = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "bm25_instance": self.bm25
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads a serialized BM25 index and source chunks from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BM25 index file not found at: {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        serialized_chunks = data.get("chunks", [])
        self.chunks = [PolicyChunk.from_dict(c) for c in serialized_chunks]
        self.bm25 = data.get("bm25_instance")
        logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks from {filepath}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PolicyChunk, float]]:
        """Retrieves top_k chunks matching the query, with their score."""
        if self.bm25 is None:
            logger.warning("BM25 index is not initialized. Returning empty list.")
            return []

        query_tokens = tokenize_text(query)
        # Calculate BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Pair chunk, score
        pairs = list(zip(self.chunks, scores))
        # Sort descending by score
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter down to top_k and positive scores if possible, though BM25 can be 0 or small
        results = pairs[:top_k]
        logger.info(f"Retrieved {len(results)} chunks via BM25 for query: '{query}'")
        return results
