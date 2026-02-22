import os
import requests
import time # ADDED: Essential for free-tier retry logic
from typing import List, Dict, Any
from fastapi import HTTPException, status

class VectorStoreService:
    def __init__(self):
        # ... (Your existing API Key / Host setup remains here) ...
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_host = os.getenv("PINECONE_HOST") # PINECONE_HOST is required for 2026 Direct API
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set in environment")
        if not self.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set in environment")  
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        hf_headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
        
        # FREE TIER MOD: We send the WHOLE list at once. 
        # This counts as 1 request instead of 10+, saving your free quota.
        payload = {
            "inputs": texts, 
            "options": {"wait_for_model": True} # ADDED: Wakes up the model if it's 'sleeping'
        }
        
        # ADDED: Retry loop. Free APIs sometimes flicker; this "heals" the connection.
        for attempt in range(3):
            try:
                response = requests.post(self.hf_url, headers=hf_headers, json=payload, timeout=20)
                
                if response.status_code == 429: # Rate limit hit
                    time.sleep(2 ** attempt) # Wait a second and try again
                    continue
                    
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == 2:
                    raise HTTPException(status_code=502, detail=f"HuggingFace Busy: {str(e)}")
        return []

    def store_vectors(self, material_id: int, chunks: List[str]) -> bool:
        try:
            # FREE TIER MOD: Logic shifted to batch processing.
            embeddings = self.get_embeddings(chunks)
            
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append({
                    "id": f"mat_{material_id}_chunk_{i}",
                    "values": embedding,
                    "metadata": {"text": chunk, "material_id": material_id}
                })
            
            # Pinecone Upsert
            headers = {"Api-Key": self.pinecone_api_key, "Content-Type": "application/json"}
            response = requests.post(f"{self.pinecone_host}/vectors/upsert", headers=headers, json={"vectors": vectors})
            response.raise_for_status()
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pinecone Sync Failed: {str(e)}")