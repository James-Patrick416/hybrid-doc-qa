from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from src.retrieval import get_chunks, HybridRetriever

app = FastAPI(title="Hybrid Document QA API", version="1.0")

# Initialize chunks and retriever on startup
chunks = get_chunks()
retriever = HybridRetriever(chunks)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.get("/")
def health_check():
    return {"status": "healthy", "chunks_loaded": len(chunks)}

@app.post("/qa")
def answer_question(req: QueryRequest):
    try:
        # 1. Retrieve top chunks using hybrid search + reranker
        results = retriever.search(req.query, top_k=req.top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant context found.")
            
        context = "\n\n".join([res["text"] for res in results])
        
        # 2. Construct grounded prompt
        prompt = f"""You are a precise assistant. Answer the question using ONLY the provided context below. If the answer cannot be found, say "I cannot answer this based on the provided documents."

Context:
{context}

Question: {req.query}
Answer:"""

        # 3. Query local Ollama instance (ensure Ollama is accessible)
        response = requests.post(
            "http://host.docker.internal:11434/api/generate" if "DOCKER_CONTAINER" in os.environ else "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Local LLM generation failed.")
            
        answer = response.json().get("response", "").strip()
        
        return {
            "query": req.query,
            "answer": answer,
            "retrieved_chunks": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
