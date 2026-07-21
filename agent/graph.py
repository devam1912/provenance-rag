import os
import json
import re
import logging
import time
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
from grounding.citation_validator import validate_citations

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
    """Initializes the LLM with timeouts, max_retries, and fallback options."""
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        if provider == "google":
            model_name = "gemini-3.5-flash"
        elif provider == "mistral":
            model_name = "mistral-large-latest"
        else:
            model_name = "gpt-4o-mini"
            
    if provider == "mistral" and ("gemini" in model_name or not model_name):
        model_name = "mistral-large-latest"
        
    timeout = float(os.getenv("LLM_TIMEOUT", "10.0"))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
    
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=google.")
        
        primary_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        
        fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gemini-1.5-flash")
        if fallback_model != model_name:
            fallback_llm = ChatGoogleGenerativeAI(
                model=fallback_model,
                temperature=0.0,
                google_api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
            return primary_llm.with_fallbacks([fallback_llm])
            
        return primary_llm
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        primary_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            timeout=timeout,
            max_retries=max_retries
        )
        fallback_model = os.getenv("LLM_FALLBACK_MODEL", "gpt-4o-mini")
        if fallback_model != model_name:
            fallback_llm = ChatOpenAI(
                model=fallback_model,
                temperature=0.0,
                timeout=timeout,
                max_retries=max_retries
            )
            return primary_llm.with_fallbacks([fallback_llm])
        return primary_llm
    elif provider == "mistral":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required when LLM_PROVIDER=mistral.")
        primary_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            openai_api_key=api_key,
            openai_api_base="https://api.mistral.ai/v1",
            timeout=timeout,
            max_retries=max_retries
        )
        fallback_model = os.getenv("LLM_FALLBACK_MODEL", "mistral-large-latest")
        if fallback_model != model_name:
            fallback_llm = ChatOpenAI(
                model=fallback_model,
                temperature=0.0,
                openai_api_key=api_key,
                openai_api_base="https://api.mistral.ai/v1",
                timeout=timeout,
                max_retries=max_retries
            )
            return primary_llm.with_fallbacks([fallback_llm])
        return primary_llm
    else:
        logger.warning(f"Unsupported LLM provider '{provider}'. Falling back to Google Gemini.")
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.0,
            timeout=timeout,
            max_retries=max_retries
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


def reload_retrievers():
    """Resets the BM25 and Vector retriever singletons to load newly created index files."""
    global _bm25, _vector
    _bm25 = None
    _vector = None
    logger.info("Cleared BM25 and Vector retriever singletons for reloading.")


# --- GRAPH NODES ---

def heuristic_route(query: str) -> Dict[str, Any]:
    """Offline regex-based routing to bypass API calls during tests/evaluation."""
    q = query.lower()
    
    # Check tool call
    has_gpa = "gpa" in q
    has_credits = "credit" in q or "credits" in q
    # find all floats/integers in query
    numbers = [float(x) for x in re.findall(r"\d+\.?\d*", q)]
    
    if len(numbers) >= 2 and (has_gpa or has_credits):
        gpa = 2.0
        local_credits = 0.0
        cc_transfer = 0.0
        four_year_transfer = 0.0
        
        # GPA is usually a float <= 4.0
        gpa_candidates = [n for n in numbers if n <= 4.0]
        credit_candidates = [n for n in numbers if n > 4.0]
        
        if gpa_candidates:
            gpa = gpa_candidates[0]
            
        if len(credit_candidates) >= 1:
            local_credits = credit_candidates[0]
        if len(credit_candidates) >= 2:
            cc_transfer = credit_candidates[1]
        if len(credit_candidates) >= 3:
            four_year_transfer = credit_candidates[2]
            
        return {
            "route": "tool_call",
            "tool_input": {
                "gpa": gpa,
                "local_credits": local_credits,
                "community_college_transfer": cc_transfer,
                "four_year_transfer": four_year_transfer
            },
            "clarification_message": "",
            "retry_count": 0,
            "failed_citations": []
        }
        
    # Check clarify
    words = [w for w in q.split() if w.strip()]
    if len(words) <= 3 and any(w in ["probation", "warning", "rules", "credits", "standing", "limit"] for w in words):
        return {
            "route": "clarify",
            "tool_input": {},
            "clarification_message": "Could you please clarify what specific academic standing or credit transfer rule you are asking about?",
            "retry_count": 0,
            "failed_citations": []
        }
        
    # Check compound queries
    if re.search(r"\band\b", q) or "as well as" in q or "?" in q[:-1] or len(q) > 80:
        return {
            "route": "sub_questions",
            "tool_input": {},
            "clarification_message": "",
            "retry_count": 0,
            "failed_citations": []
        }
        
    return {
        "route": "direct_retrieval",
        "tool_input": {},
        "clarification_message": "",
        "retry_count": 0,
        "failed_citations": []
    }


def route_query_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates the query using JSON formatting and routes it."""
    query = state["query"]
    logger.info(f"Routing query: '{query}'")
    
    if os.getenv("OFFLINE_EVAL", "false").lower() == "true":
        logger.info("OFFLINE_EVAL enabled. Running heuristic router.")
        return heuristic_route(query)
    
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
        logger.error(f"Router failed: {e}. Falling back to offline heuristic router.")
        return heuristic_route(query)


def retrieve_chunks_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves document chunks using the fused hybrid pipeline."""
    query = state["query"]
    logger.info(f"Executing direct retrieval for query: '{query}'")
    
    bm25, vector, reranker = get_retrievers()
    filter_doc = state.get("filter_document")
    
    # Run sparse and dense channels passing the filter parameter directly to retrievers
    sparse_res = bm25.retrieve(query, top_k=30, filter_document=filter_doc)
    dense_res = vector.retrieve(query, top_k=30, filter_document=filter_doc)
    
    # RRF Fusion
    fused = reciprocal_rank_fusion(sparse_res, dense_res, k=60)
    candidate_chunks = [
        chunk for chunk, _ in fused 
        if "table of contents" not in chunk.text.lower() and "srl  topic" not in chunk.text.lower()
    ]
    
    # Reranking
    reranked = reranker.rerank(query, candidate_chunks, top_k=5)
    final_chunks = [chunk for chunk, _ in reranked]
    
    logger.info(f"Direct retrieval successfully fetched {len(final_chunks)} chunks: {[c.chunk_id for c in final_chunks]}")
    return {"retrieved_chunks": final_chunks}


def heuristic_decompose(query: str) -> List[str]:
    """Offline split of compound query by 'and' or '?' boundaries to generate sub-queries."""
    parts = re.split(r"\band\b|\?", query)
    sub_queries = [p.strip() for p in parts if p.strip()]
    return sub_queries if sub_queries else [query]


def decompose_query_node(state: AgentState) -> Dict[str, Any]:
    """Decomposes a compound query into simpler sub-queries and aggregates searches."""
    query = state["query"]
    logger.info(f"Decomposing query: '{query}'")
    
    sub_queries = []
    if os.getenv("OFFLINE_EVAL", "false").lower() == "true":
        logger.info("OFFLINE_EVAL enabled. Running heuristic decomposition.")
        sub_queries = heuristic_decompose(query)
    else:
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
            logger.error(f"Failed to decompose query: {e}. Falling back to offline heuristic decomposition.")
            sub_queries = heuristic_decompose(query)

    # Run retrieval pipeline on each sub-query
    bm25, vector, reranker = get_retrievers()
    aggregated_chunks: List[Any] = []
    seen_ids = set()
    filter_doc = state.get("filter_document")
    
    for sub_q in sub_queries:
        logger.info(f"Retrieving sub-query: '{sub_q}'")
        sparse_res = bm25.retrieve(sub_q, top_k=15, filter_document=filter_doc)
        dense_res = vector.retrieve(sub_q, top_k=15, filter_document=filter_doc)
        
        # RRF Fusion
        fused = reciprocal_rank_fusion(sparse_res, dense_res, k=30)
        for chunk, _ in fused:
            if chunk.chunk_id not in seen_ids:
                if "table of contents" not in chunk.text.lower() and "srl  topic" not in chunk.text.lower():
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
    """Clarify node setting the clarification response."""
    logger.info("Directing user to clarify query node.")
    return {"response": state.get("clarification_message", "Could you please clarify your question?")}


def synthesize_response_node(state: AgentState) -> Dict[str, Any]:
    """Generates a cited response from Gemini based on the retrieved policy chunks."""
    route = state["route"]
    query = state["query"]
    logger.info("Executing response synthesis node.")
    
    if os.getenv("OFFLINE_EVAL", "false").lower() == "true":
        logger.info("OFFLINE_EVAL enabled. Returning mock response.")
        return {"response": "Mock offline response.", "retry_count": state.get("retry_count", 0), "failed_citations": []}
        
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
        err_str = str(e).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
            return {"response": "Error: Gemini API rate limit or quota exceeded (429 Resource Exhausted). Please wait a moment or verify your API key's limits. For offline testing, you can set OFFLINE_EVAL=true in your .env file to run without hitting the API."}
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


def validate_citations_node(state: AgentState) -> Dict[str, Any]:
    """Validates the citations in the generated response."""
    route = state["route"]
    
    if os.getenv("OFFLINE_EVAL", "false").lower() == "true":
        logger.info("OFFLINE_EVAL enabled. Bypassing citation validation.")
        return {"failed_citations": [], "retry_count": state.get("retry_count", 0)}
        
    # We only validate responses that underwent retrieval (direct_retrieval or sub_questions)
    if route not in ["direct_retrieval", "sub_questions"]:
        logger.info("Skipping citation validation because query was not resolved via retrieval.")
        return {"failed_citations": [], "retry_count": state.get("retry_count", 0)}
        
    answer = state["response"]
    chunks = state["retrieved_chunks"]
    
    # Special fallback check - if the answer is the fallback string or rate limit message, skip validation to avoid loops
    if ("insufficient verified information" in answer.lower() or 
        "insufficient information" in answer.lower() or
        "quota exceeded" in answer.lower() or
        "rate limit" in answer.lower() or
        "resource_exhausted" in answer.lower()):
        logger.info("Response is fallback or rate limit message. Skipping citation validation.")
        return {"failed_citations": [], "retry_count": state.get("retry_count", 0)}
        
    try:
        llm = get_llm()
        result = validate_citations(answer, chunks, llm)
        
        current_retry = state.get("retry_count", 0)
        if not result["is_valid"]:
            new_retry = current_retry + 1
            logger.warning(f"Citation validation FAILED on attempt {new_retry}. Failures: {result['failed_citations']}")
            
            ret_dict = {
                "failed_citations": result["failed_citations"],
                "retry_count": new_retry
            }
            
            # If max retries is reached (attempt 2), override the response with clean fallback
            if new_retry >= 2:
                logger.warning("Max retries reached. Overriding response with clean fallback.")
                ret_dict["response"] = "Insufficient verified information is available to answer the query."
                ret_dict["failed_citations"] = []  # Clear to route to end
                
            return ret_dict
        else:
            logger.info("Citation validation PASSED.")
            return {
                "failed_citations": [],
                "retry_count": current_retry
            }
    except Exception as e:
        logger.error(f"Error during citation validation node execution: {e}. Passing through.")
        return {
            "failed_citations": [],
            "retry_count": state.get("retry_count", 0)
        }


def should_retry_edge(state: AgentState) -> str:
    """Decides whether to retry synthesis or end execution based on validation results."""
    failed = state.get("failed_citations", [])
    retry = state.get("retry_count", 0)
    
    if not failed:
        logger.info("Validation passed. Routing to END.")
        return "end"
    elif retry >= 2:
        logger.warning(f"Max validation retries reached ({retry}). Routing to END.")
        return "end"
    else:
        logger.info(f"Routing back to synthesize for retry attempt {retry + 1} due to validation failures.")
        return "retry"


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
    workflow.add_node("validate", validate_citations_node)
    
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
    workflow.add_edge("synthesize", "validate")
    
    # Conditional edge from validate
    workflow.add_conditional_edges(
        "validate",
        should_retry_edge,
        {
            "retry": "synthesize",
            "end": END
        }
    )
    
    # Define End Points
    workflow.add_edge("clarify", END)
    
    return workflow.compile()

# Global compiled graph instance
agent_graph = create_agent_graph()


def get_langfuse_callback():
    """Initializes the Langfuse callback handler if keys are set in environment."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if public_key and secret_key:
        try:
            from langfuse.callback import CallbackHandler
            logger.info("Initializing Langfuse CallbackHandler for graph tracing...")
            return CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse callback handler: {e}")
    return None


def invoke_agent(query: str, chat_history: List[Dict[str, Any]] = None, config: Dict[str, Any] = None, filter_document: Optional[str] = None) -> Dict[str, Any]:
    """Wraps agent graph invocation, logs telemetry locally, and routes traces to Langfuse."""
    if chat_history is None:
        chat_history = []
    if config is None:
        config = {}
        
    # Wire Langfuse callback if available
    callbacks = config.get("callbacks", [])
    lf_callback = get_langfuse_callback()
    if lf_callback:
        callbacks.append(lf_callback)
    config["callbacks"] = callbacks
    
    initial_state = {
        "query": query,
        "chat_history": chat_history,
        "route": "",
        "current_sub_queries": [],
        "retrieved_chunks": [],
        "tool_input": {},
        "tool_output": "",
        "clarification_message": "",
        "response": "",
        "retry_count": 0,
        "failed_citations": [],
        "filter_document": filter_document
    }
    
    start_time = time.time()
    status = "SUCCESS"
    try:
        final_state = agent_graph.invoke(initial_state, config=config)
    except Exception as e:
        status = "ERROR"
        logger.error(f"Agent invocation failed: {e}")
        raise e
    finally:
        elapsed = time.time() - start_time
        
        offline_eval = os.getenv("OFFLINE_EVAL", "false").lower() == "true"
        
        # Estimate tokens programmatically for local reporting
        input_words = len(query.split())
        output_words = len(final_state.get("response", "").split())
        
        chunk_words = 0
        if not offline_eval:
            for chunk in final_state.get("retrieved_chunks", []):
                chunk_words += len(chunk.text.split())
                
        input_tokens = int(input_words * 1.33 + chunk_words * 1.33 + 500)
        output_tokens = int(output_words * 1.33)
        
        # Count approximate LLM invocations based on route
        calls_count = 1
        route = final_state.get("route", "")
        if route == "sub_questions" and not offline_eval:
            calls_count += 1
        if route in ["direct_retrieval", "sub_questions"] and not offline_eval:
            retry_count = final_state.get("retry_count", 0)
            calls_count += (retry_count + 1) * 2
            
        total_input_tokens = input_tokens * calls_count
        total_output_tokens = output_tokens * (final_state.get("retry_count", 0) + 1)
        
        # Est cost based on Gemini 1.5 Flash tier rates
        cost = 0.0
        if not offline_eval:
            cost = ((total_input_tokens / 1_000_000) * 0.075) + ((total_output_tokens / 1_000_000) * 0.30)
            
        telemetry_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": query,
            "route": route,
            "latency_seconds": elapsed,
            "tokens_used": total_input_tokens + total_output_tokens,
            "estimated_cost_usd": cost,
            "status": status,
            "retrieved_chunks_count": len(final_state.get("retrieved_chunks", [])),
            "failed_citations_count": len(final_state.get("failed_citations", [])),
            "retry_count": final_state.get("retry_count", 0),
            "offline_eval": offline_eval
        }
        
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/telemetry.jsonl", "a") as f:
                f.write(json.dumps(telemetry_entry) + "\n")
        except Exception as logger_err:
            logger.error(f"Failed to write local telemetry: {logger_err}")
            
    return final_state
