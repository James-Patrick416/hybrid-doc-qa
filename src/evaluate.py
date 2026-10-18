import time
import requests
from src.retrieval import get_chunks, HybridRetriever

# Define a test dataset mapping queries to the expected chunk ID(s)
TEST_DATASET = [
    {
        "query": "What is the mechanism of action of Regular Insulin?",
        "relevant_chunk_ids": [1]
    },
    {
        "query": "What are the rapid-acting insulins?",
        "relevant_chunk_ids": [0]
    },
    {
        "query": "What are the indications and prototype for Glucocorticoids?",
        "relevant_chunk_ids": [7]
    }
]

def evaluate_retrieval(retriever):
    print("\n--- Running Retrieval Recall Evaluation ---")
    total_queries = len(TEST_DATASET)
    hits_rerank = 0
    
    latencies = []

    for item in TEST_DATASET:
        query = item["query"]
        expected_ids = item["relevant_chunk_ids"]
        
        start_time = time.time()
        results = retriever.search(query, top_k=3, candidate_pool=10)
        elapsed = (time.time() - start_time) * 1000 # ms
        latencies.append(elapsed)
        
        retrieved_ids = [res["chunk_id"] for res in results]
        
        # Check if any expected ID is in retrieved top 3
        hit = any(eid in retrieved_ids for eid in expected_ids)
        if hit:
            hits_rerank += 1
            
        print(f"Query: '{query}'")
        print(f"  Expected: {expected_ids} | Retrieved: {retrieved_ids} | Match: {hit} | Latency: {elapsed:.2f}ms")

    recall_at_3 = hits_rerank / total_queries
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    
    print(f"\n[Metrics Summary]")
    print(f"Recall@3 (Hybrid + Rerank): {recall_at_3 * 100:.1f}%")
    print(f"p95 Retrieval Latency: {p95_latency:.2f}ms")

def test_local_llm_generation(retriever):
    print("\n--- Testing Local TinyLlama Generation & Grounding ---")
    query = "What is the mechanism of action of Regular Insulin?"
    results = retriever.search(query, top_k=2)
    
    context = "\n\n".join([res["text"] for res in results])
    
    prompt = f"""You are a precise medical assistant. Answer the question using ONLY the provided context below. If the answer cannot be found, say "I cannot answer this based on the provided documents."

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            answer = response.json().get("response", "No response field")
            print(f"Prompt:\n{query}\n")
            print(f"Retrieved Context Snippet:\n{context[:300]}...\n")
            print(f"TinyLlama Generated Answer:\n{answer.strip()}")
        else:
            print(f"Ollama returned status code {response.status_code}. Is TinyLlama running?")
    except Exception as e:
        print(f"Could not connect to local Ollama instance: {e}")

if __name__ == "__main__":
    chunks = get_chunks()
    retriever = HybridRetriever(chunks)
    
    evaluate_retrieval(retriever)
    test_local_llm_generation(retriever)