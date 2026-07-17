import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Local imports
from agent.state import AgentState
from agent.tools.credit_calculator import execute_credit_calculator_tool
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.rerank import CrossEncoderReranker

# Load env configurations
load_dotenv()

logger = logging.getLogger("agent.graph")

def extract_text_content(content: Any) -> str:
    """Helper to safely extract string text from string or block-list outputs."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "".join(texts)
    return str(content)


def get_llm():
    """Initializes the Gemini model with a 30s timeout to avoid indefinite hanging."""
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash")
    
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=google.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=api_key,
            timeout=30.0  # Safe request timeout
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            timeout=30.0
        )
    else:
        logger.warning(f"Unsupported LLM provider '{provider}'. Falling back to Google Gemini.")
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.0,
            timeout=30.0
        )


# --- Global Retrievers Singleton ---
_bm25 = None
_vector = None
_reranker = None

def get_retrievers():
    """Singleton getter for BM25, Vector, and Rerank tools."""
    global _bm25, _vector, _reranker
    bm25_index_path = os.getenv("BM25_INDEX_PATH", "./data/bm25/bm25.pkl")
    chroma_db_dir = os.getenv("CHROMA_DB_DIR", "./data/chroma")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    if _bm25 is None:
        logger.info("Initializing BM25 Retriever singleton...")
        _bm25 = BM25Retriever()
        _bm25.load(bm25_index_path)
    if _vector is None:
        logger.info("Initializing Vector Retriever singleton...")
        _vector = VectorRetriever(
            db_dir=chroma_db_dir,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        _vector.load()
    if _reranker is None:
        logger.info("Initializing CrossEncoder Reranker singleton...")
        _reranker = CrossEncoderReranker(model_name=reranker_model)
        
    return _bm25, _vector, _reranker


# --- GRAPH NODES ---

def route_query_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates the query using JSON formatting and routes it."""
    query = state["query"]
    logger.info(f"Routing query: '{query}'")
    
    system_prompt = (
        "You are the routing orchestrator for a university academic policy advisor system.\n"
        "Evaluate the user's query and decide how to route it.\n\n"
        "Select one of the following execution routes:\n"
        "1. 'tool_call': Choose this ONLY if the query asks for a GPA/credit check, transfer audit, or graduation status check AND provides specific numeric academic metrics (e.g. GPA, local credits, transfer credits).\n"
        "2. 'sub_questions': Choose this if the query is a compound query containing multiple distinct questions that need separate policy retrieval steps (e.g. 'What is the community college transfer limit AND how do I apply for graduation?').\n"
        "3. 'direct_retrieval': Choose this for any clear, direct question about academic rules, limits, or procedures that can be resolved via standard document search (e.g. 'What GPA is required to be on the Dean's List?').\n"
        "4. 'clarify': Choose this if the query is extremely vague, brief, or completely lacks context (e.g. 'GPA', 'warning', 'rules', 'how do I graduate?').\n\n"
        "You MUST respond ONLY with a raw JSON object matching this schema:\n"
        "{\n"
        '  "route": "direct_retrieval" | "sub_questions" | "tool_call" | "clarify",\n'
        '  "rationale": "Brief reasoning for choosing this specific route",\n'
        '  "gpa": float | null,\n'
        '  "local_credits": float | null,\n'
        '  "community_college_transfer": float | null,\n'
        '  "four_year_transfer": float | null\n'
        "}\n"
        "Do not include any other markdown formatting outside of a potential JSON code block."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: {query}")
    ]
    
    try:
        llm = get_llm()
        res = llm.invoke(messages)
        content_str = extract_text_content(res.content).strip()
        
        # Parse JSON
        json_match = re.search(r"({.*})", content_str, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group(1))
            route = decision.get("route", "direct_retrieval")
            rationale = decision.get("rationale", "")
            logger.info(f"Router Decision -> Route: '{route}' | Rationale: '{rationale}'")
            
            tool_input = {}
            if route == "tool_call":
                tool_input = {
                    "gpa": decision.get("gpa") if decision.get("gpa") is not None else 0.0,
                    "local_credits": decision.get("local_credits") if decision.get("local_credits") is not None else 0.0,
                    "community_college_transfer": decision.get("community_college_transfer") if decision.get("community_college_transfer") is not None else 0.0,
                    "four_year_transfer": decision.get("four_year_transfer") if decision.get("four_year_transfer") is not None else 0.0,
                }
            
            return {
                "route": route,
                "tool_input": tool_input,
                "clarification_message": "Could you please clarify what specific academic standing or credit transfer rule you are asking about?" if route == "clarify" else "",
                "retry_count": 0,
                "failed_citations": []
            }
        else:
            logger.warning(f"No JSON found in response. Raw response: {content_str}")
            raise ValueError("No JSON block found.")
            
    except Exception as e:
        logger.error(f"Router failed: {e}. Defaulting to direct_retrieval.")
        return {
            "route": "direct_retrieval",
            "tool_input": {},
            "clarification_message": "",
            "retry_count": 0,
            "failed_citations": []
        }


def retrieve_chunks_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves document chunks using the fused hybrid pipeline."""
    query = state["query"]
    logger.info(f"Executing direct retrieval for query: '{query}'")
    
    bm25, vector, reranker = get_retrievers()
    
    # Run sparse and dense channels
    sparse_res = bm25.retrieve(query, top_k=30)
    dense_res = vector.retrieve(query, top_k=30)
    
    # RRF Fusion
    fused = reciprocal_rank_fusion(sparse_res, dense_res, k=60)
    candidate_chunks = [chunk for chunk, _ in fused]
    
    # Reranking
    reranked = reranker.rerank(query, candidate_chunks, top_k=5)
    final_chunks = [chunk for chunk, _ in reranked]
    
    logger.info(f"Direct retrieval successfully fetched {len(final_chunks)} chunks.")
    return {"retrieved_chunks": final_chunks}


def decompose_query_node(state: AgentState) -> Dict[str, Any]:
    """Decomposes a compound query into simpler sub-queries and aggregates searches."""
    query = state["query"]
    logger.info(f"Decomposing query: '{query}'")
    
    system_prompt = (
        "You are an academic policy decomposition engine.\n"
        "Your task is to split a complex or multi-part user query into 2 to 3 simpler, search-ready questions.\n"
        "Each generated sub-question must be focused on a single academic policy topic.\n\n"
        "You MUST respond ONLY with a raw JSON array of strings, e.g.:\n"
        '[\n'
        '  "sub-question 1",\n'
        '  "sub-question 2"\n'
        ']\n'
        "Do not include any other markdown formatting outside of a potential JSON code block."
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Complex Query: {query}")
    ]
    
    sub_queries = []
    try:
        llm = get_llm()
        res = llm.invoke(messages)
        content_str = extract_text_content(res.content).strip()
        
        json_match = re.search(r"(\[.*\])", content_str, re.DOTALL)
        if json_match:
            sub_queries = json.loads(json_match.group(1))
            logger.info(f"Generated sub-queries: {sub_queries}")
        else:
            raise ValueError("No JSON array found.")
    except Exception as e:
        logger.error(f"Failed to decompose query: {e}. Falling back to main query.")
        sub_queries = [query]

    # Run retrieval pipeline on each sub-query
    bm25, vector, reranker = get_retrievers()
    aggregated_chunks: List[Any] = []
    seen_ids = set()
    
    for sub_q in sub_queries:
        logger.info(f"Retrieving sub-query: '{sub_q}'")
        sparse_res = bm25.retrieve(sub_q, top_k=15)
        dense_res = vector.retrieve(sub_q, top_k=15)
        
        fused = reciprocal_rank_fusion(sparse_res, dense_res, k=60)
        candidate_chunks = [chunk for chunk, _ in fused]
        
        reranked = reranker.rerank(sub_q, candidate_chunks, top_k=3)
        
        for chunk, _ in reranked:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                aggregated_chunks.append(chunk)

    logger.info(f"Aggregated {len(aggregated_chunks)} unique chunks from {len(sub_queries)} sub-queries.")
    return {
        "current_sub_queries": sub_queries,
        "retrieved_chunks": aggregated_chunks
    }


def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes the credit calculator advising tool."""
    tool_input = state["tool_input"]
    logger.info(f"Executing credit calculator node with inputs: {tool_input}")
    
    output = execute_credit_calculator_tool(tool_input)
    return {"tool_output": output}


def clarify_query_node(state: AgentState) -> Dict[str, Any]:
    """Empty pass-through clarify node setting state details."""
    logger.info("Directing user to clarify query node.")
    return {}


def synthesize_response_node(state: AgentState) -> Dict[str, Any]:
    """Generates a cited response from Gemini based on the retrieved policy chunks."""
    route = state["route"]
    query = state["query"]
    logger.info("Executing response synthesis node.")
    
    if route == "clarify":
        return {"response": state["clarification_message"]}
        
    # For tool calls, ask Gemini to format the numeric results in a friendly advising tone
    if route == "tool_call":
        tool_output = state["tool_output"]
        system_prompt = (
            "You are a helpful and precise university academic policy advisor.\n"
            "The user asked a query that required credit calculator verification.\n"
            "Here is the raw numeric credit/GPA audit output:\n"
            f"{tool_output}\n\n"
            "Your task is to write a friendly, supportive, and clear advisor response explaining "
            "these audit results to the student. Maintain accuracy and do not add any citations "
            "since this is derived from direct user input metrics."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Query: {query}")
        ]
        try:
            llm = get_llm()
            res = llm.invoke(messages)
            ans = extract_text_content(res.content)
            return {"response": ans}
        except Exception as e:
            logger.error(f"Failed to synthesize tool call explanation: {e}")
            return {"response": f"Calculated credit audit results:\n\n{tool_output}"}

    # For direct retrieval or sub_questions, synthesize based on chunks
    chunks = state["retrieved_chunks"]
    if not chunks:
        return {"response": "Insufficient verified information is available to answer the query."}
        
    chunks_text = ""
    for c in chunks:
        chunks_text += f"Reference Chunk ID: {c.chunk_id}\nContent:\n{c.text}\n---\n"
        
    system_prompt = (
        "You are a helpful and precise university academic policy advisor.\n"
        "Your goal is to answer the student's question based strictly on the provided policy document chunks.\n\n"
        "CRITICAL RULES:\n"
        "1. Every factual statement or claim you make must be immediately followed by a citation pointing to the exact Reference Chunk ID it came from, formatted as `[filename#chunk_idx]`. For example: 'Undergraduate students must maintain a cumulative GPA of 2.0 or higher to remain in good academic standing [academic_standing.txt#chunk_1].'\n"
        "2. Place the citation right next to the specific sentence/fact it supports. Do NOT group citations at the end of paragraphs.\n"
        "3. Only cite the reference chunks provided. Do NOT make up any chunk IDs.\n"
        "4. Do NOT make any claims that cannot be directly supported by the text. If the references do not contain enough information to answer a part of the query, explicitly state that there is insufficient information for that part.\n"
        "5. If you cannot answer the user query at all based on the references, respond exactly with: 'Insufficient verified information is available to answer the query.'\n\n"
        "References:\n"
        f"{chunks_text}"
    )
    
    # Prepend failed citations feedback if retry turn
    if state.get("failed_citations"):
        feedback = "\n".join(state["failed_citations"])
        system_prompt += (
            f"\n\nIMPORTANT CORRECTION REQUIRED:\n"
            f"Your previous response attempt was rejected due to the following validation errors:\n"
            f"{feedback}\n"
            f"Please rewrite the response, correcting these citations and removing any claims not fully supported by the reference texts. Do not reuse any invalid citations."
        )
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Student Question: {query}")
    ]
    
    try:
        llm = get_llm()
        res = llm.invoke(messages)
        ans = extract_text_content(res.content)
        return {"response": ans}
    except Exception as e:
        logger.error(f"Failed to synthesize response: {e}")
        return {"response": "Insufficient verified information is available to answer the query."}


# --- CONDITIONAL ROUTER ---

def decide_next_node(state: AgentState) -> str:
    """Decides the conditional routing edge based on state['route']."""
    route = state["route"]
    if route == "tool_call":
        return "tool"
    elif route == "clarify":
        return "clarify"
    elif route == "sub_questions":
        return "decompose"
    else:
        return "retrieve"


# --- BUILD STATE GRAPH ---

def create_agent_graph() -> StateGraph:
    """Constructs and compiles the StateGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    workflow.add_node("router", route_query_node)
    workflow.add_node("retrieve", retrieve_chunks_node)
    workflow.add_node("decompose", decompose_query_node)
    workflow.add_node("tool", execute_tool_node)
    workflow.add_node("clarify", clarify_query_node)
    workflow.add_node("synthesize", synthesize_response_node)
    
    # Define Entry Point
    workflow.set_entry_point("router")
    
    # Define Conditional Edges
    workflow.add_conditional_edges(
        "router",
        decide_next_node,
        {
            "tool": "tool",
            "clarify": "clarify",
            "decompose": "decompose",
            "retrieve": "retrieve"
        }
    )
    
    # Define Standard Transitions
    workflow.add_edge("retrieve", "synthesize")
    workflow.add_edge("decompose", "synthesize")
    workflow.add_edge("tool", "synthesize")
    
    # Define End Points
    workflow.add_edge("clarify", END)
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()

# Global compiled graph instance
agent_graph = create_agent_graph()
