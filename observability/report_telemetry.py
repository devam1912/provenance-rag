import os
import json
import numpy as np
from typing import List, Dict, Any

def display_telemetry_report() -> None:
    telemetry_file = "data/telemetry.jsonl"
    
    print("=" * 80)
    print("AGENTIC RAG SYSTEM: TELEMETRY & OBSERVABILITY REPORT")
    print("=" * 80)
    
    if not os.path.exists(telemetry_file):
        print(f"[WARNING] Telemetry log file not found at {telemetry_file}.")
        print("Please run some queries (e.g. test_graph.py or evaluate.py) to generate data first.")
        print("=" * 80)
        return
        
    latencies: List[float] = []
    costs: List[float] = []
    tokens: List[int] = []
    routes: Dict[str, int] = {}
    success_count = 0
    total_count = 0
    offline_eval_count = 0
    
    with open(telemetry_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                total_count += 1
                latencies.append(entry.get("latency_seconds", 0.0))
                costs.append(entry.get("estimated_cost_usd", 0.0))
                tokens.append(entry.get("tokens_used", 0))
                
                route = entry.get("route", "unknown")
                routes[route] = routes.get(route, 0) + 1
                
                if entry.get("status") == "SUCCESS":
                    success_count += 1
                if entry.get("offline_eval", False):
                    offline_eval_count += 1
            except Exception as e:
                # Skip corrupt lines
                continue
                
    if total_count == 0:
        print("[WARNING] Telemetry log file is empty.")
        print("=" * 80)
        return
        
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    
    total_tokens = sum(tokens)
    total_cost = sum(costs)
    avg_cost = np.mean(costs)
    success_rate = (success_count / total_count) * 100
    
    print(f"GENERAL METRICS:")
    print(f"  - Total Queries Run:          {total_count}")
    print(f"  - Successful Executions:      {success_count} ({success_rate:.1f}%)")
    print(f"  - Offline Simulated Queries:  {offline_eval_count}")
    print("-" * 80)
    
    print(f"ROUTE FREQUENCIES:")
    for route_name, count in routes.items():
        percentage = (count / total_count) * 100
        print(f"  - '{route_name}': {count} ({percentage:.1f}%)")
    print("-" * 80)
    
    print(f"LATENCY PROFILE:")
    print(f"  - p50 Latency (Median):       {p50_latency:.3f} seconds")
    print(f"  - p95 Latency:                {p95_latency:.3f} seconds")
    print(f"  - Average Latency:            {avg_latency:.3f} seconds")
    print(f"  - Max Latency:                {max_latency:.3f} seconds")
    print("-" * 80)
    
    print(f"TOKEN & COST ANALYSIS:")
    print(f"  - Total Tokens Consumed:      {total_tokens:,}")
    print(f"  - Total Estimated Cost:       ${total_cost:.5f} USD")
    print(f"  - Average Cost Per Query:     ${avg_cost:.6f} USD")
    print(f"  *Rates calculated using standard Gemini 1.5 Flash API pricing*")
    print("=" * 80)

if __name__ == "__main__":
    display_telemetry_report()
