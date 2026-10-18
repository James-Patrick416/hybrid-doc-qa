import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi

PDF_PATH = "data/james.pdf"
CHROMA_PATH = "data/chroma_db"

def get_chunks():
    reader = PdfReader(PDF_PATH)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {i+1}]\n" + text
            
    chunk_size = 500
    overlap = 50
    chunks = []
    start = 0
    while start < len(full_text):
        chunk = full_text[start:start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

class HybridRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Chroma vector store
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name="document_chunks")
        
        # Populate Chroma if empty
        if self.collection.count() == 0:
            print("Populating Chroma vector database...")
            embeddings = self.model.encode(chunks).tolist()
            ids = [str(i) for i in range(len(chunks))]
            self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        else:
            print(f"Loaded existing Chroma DB with {self.collection.count()} chunks.")
            
        # Initialize BM25 keyword index over identical chunks
        tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=5, rrf_k=60):
        # 1. Dense Vector Search
        query_embedding = self.model.encode([query]).tolist()
        dense_results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        dense_ids = [int(doc_id) for doc_id in dense_results['ids'][0]]
        
        # 2. Sparse BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        for rank, doc_id in enumerate(dense_ids):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            
        for rank, doc_id in enumerate(bm25_top_indices):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            
        # Sort by fused RRF score descending
        sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_fused[:top_k]:
            results.append({
                "chunk_id": doc_id,
                "score": score,
                "text": self.chunks[doc_id]
            })
        return results

if __name__ == "__main__":
    chunks = get_chunks()
    retriever = HybridRetriever(chunks)
    
    test_query = "Insulin lispro"
    print(f"\nTesting Hybrid Search for query: '{test_query}'")
    results = retriever.search(test_query, top_k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Chunk ID: {res['chunk_id']}, RRF Score: {res['score']:.4f}) ---")
        print(res['text'][:300] + "...")
