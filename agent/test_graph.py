import os
import logging
from dotenv import load_dotenv

# Load env configuration
load_dotenv()

# Configure logging to show graph nodes transitions
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from agent.graph import invoke_agent

def run_test_query(query: str) -> None:
    print("\n" + "=" * 80)
    print(f"TEST QUERY: '{query}'")
    print("=" * 80)
    
    try:
        # Run graph wrapper
        final_state = invoke_agent(query, chat_history=[])
        
        # Display results
        print("\n--- STATE EXECUTION RESULTS ---")
        print(f"Determined Route: '{final_state['route']}'")
        
        if final_state["current_sub_queries"]:
            print(f"Decomposed Queries: {final_state['current_sub_queries']}")
            
        if final_state["retrieved_chunks"]:
            print(f"Retrieved Chunks: {[c.chunk_id for c in final_state['retrieved_chunks']]}")
            
        if final_state["tool_input"]:
            print(f"Tool Input: {final_state['tool_input']}")
            
        print("\nFinal Response:")
        print(final_state["response"])
        
    except Exception as e:
        print(f"Error executing graph: {e}")

if __name__ == "__main__":
    # Test all 4 paths
    print("STARTING LANGGRAPH ADVISOR GRAPH INTEGRATION TESTS")
    
    # Test Path 1: Direct Retrieval
    run_test_query("What GPA is required to be on the Dean's List?")
    
    # Test Path 2: Sub Questions Decomposition
    run_test_query("What is the community college transfer cap and how do I return to good academic standing if suspended?")
    
    # Test Path 3: Tool Call (GPA & credit audit)
    run_test_query("I have a 1.8 GPA, 45 local credits, and 75 community college transfer credits. Can I graduate?")
    
    # Test Path 4: Clarification Prompt
    run_test_query("probation rules")
