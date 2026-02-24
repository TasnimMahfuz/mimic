import os
import requests
import time
from typing import List, Dict, Any
from fastapi import HTTPException, status

class VectorStoreService:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        # FIXED: Use BAAI/bge-large-en-v1.5 for 1024-d embeddings and avoid 410 error
        self.hf_url = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")
        if not self.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set in environment")
        
        # Construct Pinecone host from index name and environment
        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "mimic-rag")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-west-2-aws")
        self.pinecone_host = f"https://{pinecone_index}-{pinecone_env}.pinecone.io"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        hf_headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        for attempt in range(3):
            try:
                response = requests.post(self.hf_url, headers=hf_headers, json=payload, timeout=20)
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if response.status_code == 503:
                    raise HTTPException(status_code=502, detail="HuggingFace model is loading, please try again later.")
                response.raise_for_status()
                return response.json()
            except HTTPException:
                raise
            except Exception as e:
                if attempt == 2:
                    raise HTTPException(status_code=502, detail=f"HuggingFace Busy: {str(e)}")
        return []

    def store_vectors(self, material_id: int, chunks: List[str]) -> bool:
        try:
            embeddings = self.get_embeddings(chunks)
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append({
                    "id": f"mat_{material_id}_chunk_{i}",
                    "values": embedding,
                    "metadata": {"text": chunk, "material_id": material_id}
                })
            headers = {"Api-Key": self.pinecone_api_key, "Content-Type": "application/json"}
            response = requests.post(f"{self.pinecone_host}/vectors/upsert", headers=headers, json={"vectors": vectors})
            response.raise_for_status()
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pinecone Sync Failed: {str(e)}")

    def retrieve_vectors(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Vectorize the query, then POST to Pinecone's /query endpoint.
        """
        try:
            query_embedding = self.get_embeddings([query])[0]
            headers = {"Api-Key": self.pinecone_api_key, "Content-Type": "application/json"}
            payload = {
                "vector": query_embedding,
                "topK": top_k,
                "includeMetadata": True
            }
            response = requests.post(f"{self.pinecone_host}/query", headers=headers, json=payload)
            response.raise_for_status()
            results = response.json()
            matches = results.get("matches", [])
            return [
                {
                    "text": match["metadata"].get("text", ""),
                    "material_id": match["metadata"].get("material_id"),
                    "score": match.get("score", 0)
                }
                for match in matches
            ]
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Pinecone Query Failed: {str(e)}")