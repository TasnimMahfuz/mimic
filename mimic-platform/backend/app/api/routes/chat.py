import logging
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.api.deps import get_current_user, get_rag_service, get_llm_service
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="")


class ChatQuery(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    context: str
    status: str = "success"


@router.post("/query")
def chat_query(
    req: ChatQuery,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    """
    Process user query with graceful degradation.
    
    Returns HTTP 200 even if vector store is down.
    
    Fallback hierarchy:
    1. Vector store retrieval
    2. Safe LLM response ("Knowledge base temporarily unavailable")
    
    Always returns: {"response": str, "context": str, "status": str}
    """
    query_text = req.query.strip()
    
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    try:
        # Step 1: Try to retrieve context from vector store
        context = rag_service.retrieve_context(query_text, db)
        
        logger.info(f"Context retrieved: {len(context)} chars")
        
        # Step 2: Generate response
        if context:
            # Generate response using retrieved context
            try:
                response = llm_service.generate_response(context, query_text)
                return ChatResponse(
                    response=response,
                    context=context,
                    status="success"
                )
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                # Fallback: return safe message
                return ChatResponse(
                    response="I encountered an issue generating a response. Please try again.",
                    context=context,
                    status="degraded"
                )
        else:
            # No context found - still return 200 with graceful message
            logger.warning(f"No context found for query: {query_text}")
            return ChatResponse(
                response=(
                    "I don't have relevant information to answer your question. "
                    "Please check the uploaded materials or ask something else."
                ),
                context="",
                status="no_context"
            )
    
    except Exception as e:
        # Catch unexpected errors - return 200 with safe message
        # (DO NOT return 5xx error)
        logger.error(f"Chat query error: {e}", exc_info=True)
        return ChatResponse(
            response=(
                "The knowledge base is temporarily unavailable. "
                "Please try again in a moment."
            ),
            context="",
            status="unavailable"
        )