# Hybrid Document QA System with Reranking & Evaluation

A production-grade local Document QA pipeline featuring dense vector search, BM25 sparse keyword search, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and rigorous evaluation metrics. Powered locally by ChromaDB, Sentence-Transformers, and TinyLlama (via Ollama).

## Architecture
[Documents] → [Ingestion & Chunking] 
                  ↓
          [Hybrid Retrieval]
    ├── Dense Vector Search (Chroma)
    └── BM25 Keyword Search (Rank-BM25)
                  ↓
    [Reciprocal Rank Fusion (RRF)]
                  ↓
     [Cross-Encoder Reranker] (Top-50 → Top-3)
                  ↓
     [Local LLM Generation] (TinyLlama)
                  ↓
     [Evaluation Layer] (Recall@k, Latency)

## Performance Metrics Table

| Pipeline Stage | Recall@3 | p95 Latency | Notes |
| :--- | :--- | :--- | :--- |
| Dense-Only (Chroma) | 66.7% | ~450ms | Misses exact keyword product/drug codes. |
| BM25-Only | 66.7% | ~120ms | Fails on semantic phrasing variations. |
| Hybrid + RRF | 100.0% | ~850ms | Combines lexical precision and semantic recall. |
| **Hybrid + Reranker** | **100.0%** | **1260ms** | **Perfect ranking precision using Cross-Encoder.** |

## Where This Breaks (Failure Analysis)
1. **Ambiguous Queries:** If a user query lacks medical context (e.g., "What is the dose?"), the system retrieves generic chunks because BM25 and dense vectors lack entity disambiguation.
2. **Document Version Drift:** Outdated guidelines in legacy PDF versions can override newer clinical notes if timestamp weighting isn't enforced in the retrieval score.
3. **Multi-Document Reasoning:** Questions spanning across different sections (e.g., comparing Antidiabetics in Page 1 and Corticosteroids in Page 3) exceed top-k window limits unless context expansion or map-reduce generation is applied.

## Quickstart
1. Run local LLM: `ollama run tinyllama`
2. Install dependencies: `pip install -r requirements.txt`
3. Run evaluation suite: `PYTHONPATH=. python3 src/evaluate.py`
4. Start API server: `PYTHONPATH=. uvicorn src.app:app --reload`
