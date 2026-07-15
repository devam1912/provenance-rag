import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Agentic RAG API",
    description="Production-grade Agentic RAG System over University Academic Policies",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input/Output Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query or question.")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=[],
        description="Previous message turn list, e.g. [{'role': 'user', 'content': '...'}]"
    )

class Citation(BaseModel):
    chunk_id: str = Field(..., description="ID of the cited text chunk (e.g. filename#idx).")
    text: str = Field(..., description="The matching excerpt from the chunk.")
    score: Optional[float] = Field(None, description="The retrieval or relevance score.")

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    latency_ms: float
    total_cost: float = Field(0.0, description="Estimated API request cost in USD.")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    logger.info("Health check endpoint accessed")
    return {"status": "ok", "service": "Agentic RAG Backend"}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Placeholder endpoint for querying the Agentic RAG system."""
    logger.info(f"Received query request: {request.query}")
    # Placeholder return
    return QueryResponse(
        query=request.query,
        answer="This is a placeholder answer. Graph orchestration will be wired in Day 3.",
        citations=[
            Citation(chunk_id="academic_standing.txt#0", text="Placeholder cited text.", score=1.0)
        ],
        latency_ms=10.5,
        total_cost=0.0
    )
