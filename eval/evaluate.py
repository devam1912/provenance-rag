import os
import json
import time
import logging
from typing import List, Dict, Any

# Enforce offline mode for evaluation runs to avoid API calls and quota exhaustion
os.environ["OFFLINE_EVAL"] = "true"

# Disable verbose debug logging from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from agent.graph import invoke_agent

def run_evaluation() -> None:
    print("=" * 80)
    print("STARTING PROGRAMMATIC RAG PIPELINE EVALUATION (OFFLINE MODE)")
    print("=" * 80)
    
    golden_set_path = "./tests/golden_set.json"
    report_path = "./eval/report.json"
    
    if not os.path.exists(golden_set_path):
        raise FileNotFoundError(f"Golden set not found at {golden_set_path}")
        
    with open(golden_set_path, "r") as f:
        test_cases = json.load(f)
        
    logger = logging.getLogger("eval.evaluate")
    logger.setLevel(logging.INFO)
    
    results: List[Dict[str, Any]] = []
    
    total_cases = len(test_cases)
    route_correct_count = 0
    total_recall = 0.0
    total_precision = 0.0
    total_latency = 0.0
    
    print(f"Loaded {total_cases} test cases from golden set.\n")
    print(f"{'No.':<4} | {'Expected Route':<16} | {'Actual Route':<16} | {'Recall':<6} | {'Precision':<9} | {'Latency':<7} | {'Query'}")
    print("-" * 100)
    
    for idx, case in enumerate(test_cases):
        query = case["query"]
        expected_route = case["expected_route"]
        expected_chunks = case["expected_chunks"]
        
        initial_state = {
            "query": query,
            "chat_history": [],
            "route": "",
            "current_sub_queries": [],
            "retrieved_chunks": [],
            "tool_input": {},
            "tool_output": "",
            "clarification_message": "",
            "response": "",
            "retry_count": 0,
            "failed_citations": []
        }
        
        start_time = time.time()
        # Invoke LangGraph workflow wrapper
        final_state = invoke_agent(query, chat_history=[])
        elapsed = time.time() - start_time
        
        actual_route = final_state["route"]
        retrieved_ids = [c.chunk_id for c in final_state.get("retrieved_chunks", [])]
        
        # Route accuracy
        route_correct = (actual_route == expected_route)
        if route_correct:
            route_correct_count += 1
            
        # Context Recall
        expected_set = set(expected_chunks)
        retrieved_set = set(retrieved_ids)
        
        if not expected_set:
            recall = 1.0
        else:
            recall = len(retrieved_set & expected_set) / len(expected_set)
            
        # Context Precision
        if not retrieved_set:
            precision = 1.0 if not expected_set else 0.0
        else:
            precision = len(retrieved_set & expected_set) / len(retrieved_set)
            
        total_recall += recall
        total_precision += precision
        total_latency += elapsed
        
        results.append({
            "index": idx + 1,
            "query": query,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "expected_chunks": expected_chunks,
            "actual_chunks": retrieved_ids,
            "context_recall": recall,
            "context_precision": precision,
            "latency": elapsed
        })
        
        print(f"{idx+1:<4} | {expected_route:<16} | {actual_route:<16} | {recall:<6.2f} | {precision:<9.2f} | {elapsed:<7.2f}s | {query[:40]}")
        
    print("-" * 100)
    
    # Calculate averages
    route_accuracy = route_correct_count / total_cases
    avg_recall = total_recall / total_cases
    avg_precision = total_precision / total_cases
    avg_latency = total_latency / total_cases
    
    # Weighted overall score
    overall_score = (0.4 * route_accuracy) + (0.3 * avg_recall) + (0.3 * avg_precision)
    
    print(f"EVALUATION SUMMARY:")
    print(f"  - Route Accuracy:        {route_accuracy * 100:.1f}%")
    print(f"  - Average Context Recall:    {avg_recall * 100:.1f}%")
    print(f"  - Average Context Precision: {avg_precision * 100:.1f}%")
    print(f"  - Average Latency:           {avg_latency:.3f} seconds")
    print(f"  - Overall Weighted Score:    {overall_score:.4f}")
    print("=" * 80)
    
    # Save detailed report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_questions": total_cases,
            "route_accuracy": route_accuracy,
            "average_context_recall": avg_recall,
            "average_context_precision": avg_precision,
            "overall_score": overall_score,
            "average_latency": avg_latency
        },
        "results": results
    }
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed evaluation report saved to {report_path}")

    # Baseline regression check
    baseline_path = "./eval/baseline.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        baseline_score = baseline_data["summary"]["overall_score"]
        print(f"\nCOMPARING TO BASELINE:")
        print(f"  - Baseline Overall Score: {baseline_score:.4f}")
        print(f"  - Current Overall Score:  {overall_score:.4f}")
        
        # 1% tolerance for minor indexing/reranking fluctuation
        threshold = baseline_score - 0.01
        if overall_score < threshold:
            print(f"\n[ERROR] REGRESSION DETECTED: Overall score {overall_score:.4f} is below threshold {threshold:.4f}.")
            import sys
            sys.exit(1)
        else:
            print("\n[SUCCESS] Baseline checks passed! No score regression detected.")
    else:
        print("\n[INFO] No baseline.json found to compare. Saving current summary as baseline.json...")
        with open(baseline_path, "w") as f:
            json.dump({"summary": report["summary"]}, f, indent=2)
            
if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        import sys
        print(f"Evaluation error: {e}")
        sys.exit(1)
