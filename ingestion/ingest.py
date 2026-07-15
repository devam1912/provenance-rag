import os
import logging
from dotenv import load_dotenv
from ingestion.chunker import PolicyChunker
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingestion.ingest")

def run_ingestion() -> None:
    """Orchestrates loading, chunking, and indexing the policy document corpus."""
    logger.info("Starting ingestion process...")

    # Get paths and parameters from environment
    corpus_dir = os.getenv("CORPUS_DIR", "./data/corpus")
    bm25_index_path = os.getenv("BM25_INDEX_PATH", "./data/bm25/bm25.pkl")
    chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./data/chroma")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if not os.path.exists(corpus_dir):
        logger.error(f"Corpus directory not found: {corpus_dir}")
        return

    # 1. Read files and chunk them
    chunker = PolicyChunker(max_chunk_size=800, overlap=100)
    all_chunks = []
    
    logger.info(f"Scanning corpus directory: {corpus_dir}")
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                logger.info(f"Chunking file: {file_path}")
                try:
                    chunks = chunker.split_document(file_path)
                    all_chunks.extend(chunks)
                    logger.info(f"Generated {len(chunks)} chunks from {file}")
                except Exception as e:
                    logger.error(f"Error chunking {file_path}: {e}")

    if not all_chunks:
        logger.warning("No chunks generated. Ingestion aborted.")
        return

    logger.info(f"Total chunks generated across all documents: {len(all_chunks)}")

    # 2. Build and save BM25 sparse index
    logger.info("Building BM25 sparse index...")
    try:
        bm25_retriever = BM25Retriever()
        bm25_retriever.build(all_chunks)
        bm25_retriever.save(bm25_index_path)
        logger.info(f"BM25 index successfully saved to {bm25_index_path}")
    except Exception as e:
        logger.error(f"Failed to build/save BM25 index: {e}")

    # 3. Build and persist Chroma vector database
    logger.info("Building Chroma vector index...")
    try:
        vector_retriever = VectorRetriever(
            db_dir=chroma_db_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        vector_retriever.build(all_chunks)
        logger.info("Chroma vector index successfully built and persisted.")
    except Exception as e:
        logger.error(f"Failed to build/persist Chroma vector database: {e}")

    logger.info("Ingestion process completed successfully.")

if __name__ == "__main__":
    run_ingestion()
