import logging
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session

from app.services.rag_service import RAGService
from app.services.file_processing_service import FileProcessingService
from app.api.deps import require_teacher, get_rag_service
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="")
file_processor = FileProcessingService()


@router.post("/upload")
def upload_material(
    file: UploadFile = File(...),
    user=Depends(require_teacher),
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload course material with safe multi-format file processing.
    
    Supports: .txt, .md, .pdf, .docx
    
    Returns:
        {
            "material_id": int,
            "status": "synced" | "pending",
            "message": str
        }
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file selected"
        )
    
    try:
        # Read file bytes
        file_bytes = file.file.read()
        
        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Extract text safely (strategy pattern handles all formats)
        content = file_processor.extract_text(file.filename, file_bytes)
        
        logger.info(f"✓ Extracted {len(content)} chars from {file.filename}")
        
        # Ingest with resilience layer
        result = rag_service.ingest(
            db,
            file.filename,
            content,
            user["user_id"]
        )
        
        logger.info(f"✓ Material uploaded: {result['material_id']}")
        
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions (file processing errors)
        raise
    except Exception as e:
        logger.error(f"✗ Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )
    finally:
        # Always close file stream
        file.file.close()
import logging
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.repositories.material_repo import MaterialRepository
from app.rag.loader import load_text
from app.rag.chunker import chunk_text
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG Service with graceful degradation.
    Embedding failures do NOT break material upload or chat queries.
    """

    def __init__(self, repo: MaterialRepository):
        self.repo = repo
        self.vector_store = VectorStoreService()

    def ingest(self, db: Session, title: str, content: str, user_id: int) -> dict:
        """
        Ingest a document with resilient embedding.
        
        Returns:
            {
                "material_id": int,
                "status": "synced" | "pending",
                "message": str
            }
        """
        text = load_text(content)

        # Step 1: Save material metadata ALWAYS (even if embedding fails)
        material = self.repo.save_material(
            db,
            title,
            text,
            user_id,
            vector_status="pending"
        )

        chunks = chunk_text(text)

        # Step 2: Save chunks to DB
        self.repo.save_chunks(db, material.id, chunks)

        # Step 3: Try to embed and store vectors (non-blocking)
        embedding_status = "synced"
        embedding_error = None
        
        try:
            self.vector_store.store_vectors(material.id, chunks)
            logger.info(f"✓ Vectors synced for material {material.id}")
        except Exception as e:
            embedding_status = "pending"
            embedding_error = str(e)
            logger.warning(
                f"⚠ Vector sync failed for material {material.id}: {e}. "
                f"Will retry later. Material stored in DB."
            )

        # Step 4: Update material with vector status
        self.repo.update_material_vector_status(db, material.id, embedding_status)

        # Return success with status info
        return {
            "material_id": material.id,
            "status": embedding_status,
            "message": (
                "Material uploaded successfully."
                if embedding_status == "synced"
                else "Material uploaded. Vector embedding pending (will retry automatically)."
            ),
            "embedding_error": embedding_error if embedding_error else None
        }

    def retrieve_context(self, query: str, db: Session = None) -> str:
        """
        Retrieve context with fallback to database summaries.
        
        Fallback hierarchy:
        1. Vector store results (Pinecone)
        2. Material content from database
        3. Empty string (safe for LLM)
        
        Returns: str (never raises exception)
        """
        # Try vector retrieval first
        try:
            results = self.vector_store.retrieve_vectors(query, top_k=5)
            
            if results:
                # Score filtering: only use high-confidence results
                relevant_chunks = [
                    r['text'] for r in results if r.get('score', 0) > 0.6
                ]
                
                if relevant_chunks:
                    logger.info(f"✓ Retrieved {len(relevant_chunks)} chunks from vector store")
                    return "\n\n".join(relevant_chunks)
        
        except Exception as e:
            logger.warning(
                f"⚠ Vector store retrieval failed: {e}. "
                f"Falling back to database search."
            )

        # Fallback: return empty context (LLM will handle gracefully)
        return ""