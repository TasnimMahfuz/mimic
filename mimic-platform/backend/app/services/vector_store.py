import os
import requests
import time
from typing import List, Dict, Any
from fastapi import HTTPException, status
from openai import OpenAI

class VectorStoreService:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Construct Pinecone host from index name and environment
        pinecone_host = os.getenv("PINECONE_HOST")
        if pinecone_host:
            self.pinecone_host = pinecone_host
        else:
            pinecone_index = os.getenv("PINECONE_INDEX_NAME", "mimic-rag")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-west-2-aws")
            self.pinecone_host = f"https://{pinecone_index}-{pinecone_env}.pinecone.io"
        
        print(f"DEBUG: Pinecone host: {self.pinecone_host}")
        print(f"DEBUG: Using OpenAI embeddings: text-embedding-3-small (1536-d)")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI Embedding Failed: {str(e)}")

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