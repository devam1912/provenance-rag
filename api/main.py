import logging
import os
import re
import time
import io
import pypdf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Local imports
from agent.graph import invoke_agent, reload_retrievers
from ingestion.ingest import run_ingestion

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

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input/Output Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query or question.")
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        description="Previous message turn list, e.g. [{'role': 'user', 'content': '...'}]"
    )

class Citation(BaseModel):
    chunk_id: str = Field(..., description="ID of the cited text chunk (e.g. filename#idx).")
    text: str = Field(..., description="The matching excerpt from the chunk.")
    score: Optional[float] = Field(None, description="The relevance score.")

class QueryResponse(BaseModel):
    query: str
    answer: str
    route: str
    decomposed_queries: List[str]
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
    """Executes the Agentic RAG graph on the query request."""
    logger.info(f"Received query request: {request.query}")
    
    start_time = time.time()
    try:
        final_state = invoke_agent(request.query, chat_history=request.chat_history)
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing agent RAG graph: {str(e)}")
        
    elapsed_ms = (time.time() - start_time) * 1000
    
    response_text = final_state.get("response", "")
    retrieved_chunks = final_state.get("retrieved_chunks", [])
    route = final_state.get("route", "direct_retrieval")
    decomposed = final_state.get("current_sub_queries", [])
    
    # Parse citations out of the text response
    citations = []
    cited_ids = re.findall(r"\[([^\]]+#[^\]]+)\]", response_text)
    
    seen_ids = set()
    dedup_cited_ids = []
    for cid in cited_ids:
        if cid not in seen_ids:
            seen_ids.add(cid)
            dedup_cited_ids.append(cid)
            
    # Map retrieved chunks by their ID
    chunk_map = {c.chunk_id: c for c in retrieved_chunks}
    for cid in dedup_cited_ids:
        if cid in chunk_map:
            citations.append(Citation(
                chunk_id=cid,
                text=chunk_map[cid].text
            ))
        else:
            citations.append(Citation(
                chunk_id=cid,
                text="Reference content is unavailable."
            ))
            
    # Approximate or retrieve log cost
    cost = 0.0
    # Telemetry logging writes to local jsonl which contains estimated cost, but for the API, 
    # we can approximate it or return 0.0 for free Gemini tier.
    
    return QueryResponse(
        query=request.query,
        answer=response_text,
        route=route,
        decomposed_queries=decomposed,
        citations=citations,
        latency_ms=elapsed_ms,
        total_cost=cost
    )


@app.get("/api/documents")
async def list_documents():
    """Scans the data/corpus directory and returns metadata for text and markdown documents."""
    corpus_dir = os.getenv("CORPUS_DIR", "./data/corpus")
    if not os.path.exists(corpus_dir):
        return []
    
    docs = []
    try:
        for file in os.listdir(corpus_dir):
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(corpus_dir, file)
                stat = os.stat(file_path)
                docs.append({
                    "name": file,
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime
                })
    except Exception as e:
        logger.error(f"Error listing documents in corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    return docs


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts document uploads (PDF, txt, md), extracts content, saves to corpus, and triggers re-indexing."""
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename provided.")
        
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Supported extensions are .pdf, .txt, .md only.")
        
    corpus_dir = os.getenv("CORPUS_DIR", "./data/corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    
    try:
        content_bytes = await file.read()
        
        # 1. Parse text depending on file type
        if ext == ".pdf":
            pdf_file = io.BytesIO(content_bytes)
            reader = pypdf.PdfReader(pdf_file)
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            if not text_parts:
                raise HTTPException(status_code=400, detail="No readable text could be extracted from the uploaded PDF.")
                
            full_text = "\n\n".join(text_parts)
            dest_name = os.path.splitext(filename)[0] + ".txt"
            dest_path = os.path.join(corpus_dir, dest_name)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(full_text)
        else:
            dest_path = os.path.join(corpus_dir, filename)
            text_content = content_bytes.decode("utf-8", errors="ignore")
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(text_content)
                
        # 2. Trigger programmatic index ingestion pipeline
        logger.info(f"Re-indexing policy corpus database after upload: {filename}")
        run_ingestion()
        
        # 3. Clear retriever caching singletons
        reload_retrievers()
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error handling upload for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process and index document: {e}")
        
    return {"status": "success", "message": f"File {filename} uploaded, parsed, and indexed successfully."}


# Serve static frontend files from /frontend mounting at root "/"
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    logger.info(f"Mounting static files from {frontend_dir}")
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    logger.warning(f"Frontend folder not found at {frontend_dir}. Static files mount skipped.")
