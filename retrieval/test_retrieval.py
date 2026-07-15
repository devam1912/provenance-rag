import os
import logging
from dotenv import load_dotenv
from ingestion.chunker import PolicyChunk
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.rerank import CrossEncoderReranker

# Load env configurations
load_dotenv()

# Configure logging to show info in console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("retrieval.test_retrieval")

def run_retrieval_test(query: str) -> None:
    """Executes search channels and displays the retrieval pipeline comparison."""
    print("=" * 80)
    print(f"TEST QUERY: '{query}'")
    print("=" * 80)

    # 1. Initialize retrievers
    bm25_index_path = os.getenv("BM25_INDEX_PATH", "./data/bm25/bm25.pkl")
    chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./data/chroma")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Load BM25
    logger.info("Loading BM25 index...")
    bm25 = BM25Retriever()
    bm25.load(bm25_index_path)

    # Load Vector
    logger.info("Loading Vector DB index...")
    vector = VectorRetriever(
        db_dir=chroma_db_dir,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )
    vector.load()

    # Load Reranker
    logger.info("Loading Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(model_name=reranker_model)

    # 2. Retrieve Channels
    print("\n--- 1. BM25 SPARSE RETRIEVAL ---")
    bm25_results = bm25.retrieve(query, top_k=10)
    for i, (chunk, score) in enumerate(bm25_results):
        print(f"[{i+1}] ID: {chunk.chunk_id} | BM25 Score: {score:.4f}")
        print(f"    Snippet: {chunk.text[:120]}...\n")

    print("\n--- 2. CHROMA DENSE VECTOR RETRIEVAL ---")
    vector_results = vector.retrieve(query, top_k=10)
    for i, (chunk, score) in enumerate(vector_results):
        # lower is closer for Chroma L2 distance, display directly
        print(f"[{i+1}] ID: {chunk.chunk_id} | Vector Distance: {score:.4f}")
        print(f"    Snippet: {chunk.text[:120]}...\n")

    # 3. Reciprocal Rank Fusion (RRF)
    print("\n--- 3. RECIPROCAL RANK FUSION (RRF) COMBINED (Top 10) ---")
    fused_results = reciprocal_rank_fusion(bm25_results, vector_results, k=60)
    for i, (chunk, score) in enumerate(fused_results[:10]):
        print(f"[{i+1}] ID: {chunk.chunk_id} | RRF Score: {score:.6f}")
        print(f"    Snippet: {chunk.text[:120]}...\n")

    # 4. Rerank top candidate list
    print("\n--- 4. CROSS-ENCODER RERANKED (Top 5) ---")
    candidate_chunks = [chunk for chunk, _ in fused_results]
    reranked_results = reranker.rerank(query, candidate_chunks, top_k=5)
    for i, (chunk, score) in enumerate(reranked_results):
        print(f"[{i+1}] ID: {chunk.chunk_id} | Reranker Score: {score:.4f}")
        print(f"    Snippet: {chunk.text[:180]}...\n")

if __name__ == "__main__":
    # Multi-part query involving GPA thresholds and probation
    test_query = "What cumulative GPA do I need to keep, and what happens on academic probation?"
    run_retrieval_test(test_query)
