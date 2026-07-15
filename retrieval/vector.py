import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from ingestion.chunker import PolicyChunk

logger = logging.getLogger("retrieval.vector")

class VectorRetriever:
    """Dense vector retriever using Chroma DB and configurable embeddings (local/OpenAI)."""

    def __init__(
        self,
        db_dir: str = "./data/chroma",
        embedding_provider: str = "local",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "academic_policies"
    ):
        self.db_dir = db_dir
        self.embedding_provider = embedding_provider.lower()
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embeddings = self._init_embeddings()
        self.db: Optional[Chroma] = None

    def _init_embeddings(self) -> Embeddings:
        """Initializes the configured embedding model."""
        if self.embedding_provider == "openai":
            logger.info(f"Initializing OpenAI Embeddings with model: {self.embedding_model}")
            return OpenAIEmbeddings(model=self.embedding_model)
        else:
            logger.info(f"Initializing Local HuggingFace Embeddings with model: {self.embedding_model}")
            # Use local sentence-transformers
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"}
            )

    def build(self, chunks: List[PolicyChunk]) -> None:
        """Builds a new Chroma vector database from the provided chunks."""
        if not chunks:
            logger.warning("Empty chunk list provided to build vector database.")
            return

        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_dict()["metadata"] for chunk in chunks]
        # Store the chunk_id explicitly in metadata so we can map it back
        for i, meta in enumerate(metadatas):
            meta["chunk_id"] = chunks[i].chunk_id

        # Re-initialize/create new collection
        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.db_dir,
            collection_name=self.collection_name
        )
        logger.info(f"Successfully built vector database with {len(chunks)} chunks persisted to {self.db_dir}.")

    def load(self) -> None:
        """Loads an existing Chroma vector database from disk."""
        if not os.path.exists(self.db_dir):
            logger.warning(f"Chroma DB directory {self.db_dir} does not exist yet. You must build it first.")
        
        self.db = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        logger.info(f"Loaded existing vector database from {self.db_dir}.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[PolicyChunk, float]]:
        """Retrieves top_k chunks matching the query using similarity search with score."""
        if self.db is None:
            # Try to load if not initialized
            try:
                self.load()
            except Exception as e:
                logger.error(f"Failed to auto-load vector DB: {e}. Returning empty results.")
                return []

        # similarity_search_with_relevance_scores returns (Document, score)
        # Note: Chroma distance scoring can be L2 or cosine. cosine returns 0 to 1 range typically.
        # similarity_search_with_score returns (Document, distance) where distance is L2 (lower is closer).
        # We use similarity_search_with_score for robustness, lower score means more similar for default Chroma (L2).
        results = self.db.similarity_search_with_score(query, k=top_k)
        
        retrieved_pairs = []
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id", "unknown_id")
            # Map back to PolicyChunk object
            chunk = PolicyChunk(
                chunk_id=chunk_id,
                text=doc.page_content,
                source=doc.metadata.get("source", ""),
                headers=doc.metadata.get("headers", []),
                chunk_index=doc.metadata.get("chunk_index", 0)
            )
            retrieved_pairs.append((chunk, float(score)))

        logger.info(f"Retrieved {len(retrieved_pairs)} chunks via vector search for query: '{query}'")
        return retrieved_pairs
