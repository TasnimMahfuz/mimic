from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.api.deps import get_current_user
from app.db.session import get_db 


router = APIRouter(prefix="")

rag_service = RAGService()
llm_service = LLMService()


class ChatQuery(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    context: str



@router.post("/query")
def chat_query(
    req: ChatQuery,
    user=Depends(get_current_user),
    db: Session = Depends(get_db)
) -> ChatResponse:
    """
    Process a user query: retrieve context from Pinecone and generate response via Groq.
    """
    try:
        # Retrieve context from vector store
        context = rag_service.retrieve_context(req.query)
        
        if "Error" in context or "No relevant" in context:
            return ChatResponse(
                response="Unable to find relevant context for your query.",
                context=context
            )
        
        # Generate response using LLM
        response = llm_service.generate_response(context, req.query)
        
        return ChatResponse(
            response=response,
            context=context
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )