import logging
from fastapi import HTTPException, status
from app.repositories.material_repo import MaterialRepository
from app.rag.loader import load_text
from app.rag.chunker import chunk_text
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

class RAGService:

    def __init__(self):
        self.repo = MaterialRepository()
        self.vector_store = VectorStoreService()

    def ingest(self, db, title, content, user_id):
        """
        Ingest a document: extract text, chunk it, embed via cloud API, store in Pinecone.
        """
        text = load_text(content)

        material = self.repo.save_material(
            db,
            title,
            text,
            user_id
        )

        chunks = chunk_text(text)

        # Save chunks to DB
        self.repo.save_chunks(
            db,
            material.id,
            chunks
        )

        # Store vectors in Pinecone (cloud embeddings)
        try:
            self.vector_store.store_vectors(material.id, chunks)
        except Exception as e:
            print(f"Warning: Failed to store vectors in Pinecone: {e}")
            logger.error(f"Pinecone Sync Failed: {e}")
            # Don't fail the entire ingest if vector storage fails
            # Material is still saved in DB

        return material.id

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context from Pinecone for a given query.
        """
        try:
            results = self.vector_store.retrieve_vectors(query, top_k=5)
            
            if not results:
                return ""  # No relevant context found
            
            # ADDED: Score filtering
            # WHY: Pinecone might return irrelevant text with low scores. 
            # We only want the high-quality stuff for the LLM.
            relevant_chunks = [
                r['text'] for r in results if r.get('score', 0) > 0.6
            ]
            
            if not relevant_chunks:
                return "No relevant information found in the documents."

            # MODIFIED: Removed the [Score: 0.xx] from the string
            # WHY: The LLM doesn't need to see technical scores; it just needs the text.
           
            return "\n\n".join(relevant_chunks)
        except Exception as e:
            logger.error(f"Vector Retrieval Error: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Knowledge base unreachable"
            )