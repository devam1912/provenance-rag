from typing import TypedDict, List, Dict, Any, Optional
from ingestion.chunker import PolicyChunk

class AgentState(TypedDict):
    """
    State definition for the LangGraph orchestrator.
    
    Attributes:
        query: The raw query input from the user.
        chat_history: List of conversational turns (e.g. [{'role': 'user', 'content': '...'}]).
        route: The execution path determined by the router ('direct_retrieval', 'sub_questions', 'tool_call', 'clarify').
        current_sub_queries: A list of sub-questions generated during query decomposition.
        retrieved_chunks: A list of unique PolicyChunk objects retrieved from the corpus.
        tool_input: Parsed numeric and text arguments mapped to the credit validator tool.
        tool_output: Plain text summary output from the credit calculator tool execution.
        clarification_message: A message prompting the user to clarify their request.
        response: The final generated response text.
    """
    query: str
    chat_history: List[Dict[str, str]]
    route: str
    current_sub_queries: List[str]
    retrieved_chunks: List[PolicyChunk]
    tool_input: Dict[str, Any]
    tool_output: str
    clarification_message: str
    response: str
