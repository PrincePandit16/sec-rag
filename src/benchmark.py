import asyncio
import time
import httpx
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/query"
QUERIES = [
    {"query": "What was NVIDIA's net income in 2024?", "agentic": True},
    {"query": "What is Apple's total revenue in 2023?", "agentic": True},
    {"query": "Does Microsoft mention any new AI investments in their 2023 10-K?", "agentic": True},
    {"query": "Search for 'consolidated statements of operations' for Alphabet 2023", "agentic": False},
    {"query": "What is the company name in the first document?", "agentic": False},
]

async def measure_query(client, query_data):
    start = time.perf_counter()
    try:
        response = await client.post(API_URL, json=query_data, timeout=120.0)
        end = time.perf_counter()
        if response.status_code == 200:
            return {
                "query": query_data["query"],
                "latency": end - start,
                "status": "Success",
                "agentic": query_data["agentic"]
            }
        else:
            return {"query": query_data["query"], "latency": end - start, "status": f"Error {response.status_code}", "agentic": query_data["agentic"]}
    except Exception as e:
        end = time.perf_counter()
        return {"query": query_data["query"], "latency": end - start, "status": f"Exception: {str(e)}", "agentic": query_data["agentic"]}

async def run_benchmark():
    print("🚀 Starting SEC RAG Latency Benchmark...")
    print(f"Target API: {API_URL}\n")

    async with httpx.AsyncClient() as client:
        tasks = [measure_query(client, q) for q in QUERIES]
        results = await asyncio.gather(*tasks)

    # Analysis
    print(f"{'Query':<60} | {'Type':<10} | {'Status':<10} | {'Latency':<10}")
    print("-" * 95)

    total_latency = 0
    success_count = 0

    for res in results:
        print(f"{res['query'][:58]:<60} | {'Agentic' if res['agentic'] else 'Simple':<10} | {res['status']:<10} | {res['latency']:.2f}s")
        if res["status"] == "Success":
            success_count += 1
            total_latency += res["latency"]

    if success_count > 0:
        print("-" * 95)
        print(f"Average Latency (Successful): {total_latency / success_count:.2f}s")
        print(f"Success Rate: {success_count}/{len(QUERIES)}")
    else:
        print("\n❌ No successful queries. Is the server running?")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
