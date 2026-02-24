from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import jwt
from sqlalchemy.orm import Session

from app.core.security import SECRET, ALGORITHM
from app.db.session import get_db
from app.repositories.material_repo import MaterialRepository
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService

security = HTTPBearer()


def get_current_user(token=Depends(security)):
    payload = jwt.decode(
        token.credentials,
        SECRET,
        algorithms=[ALGORITHM]
    )
    return payload


def require_teacher(user=Depends(get_current_user)):
    if user["role"] != "teacher":
        raise HTTPException(403, "Teacher only")
    return user


def get_material_repo(db: Session = Depends(get_db)) -> MaterialRepository:
    """Provide MaterialRepository with database session."""
    return MaterialRepository()


def get_rag_service(
    db: Session = Depends(get_db),
    repo: MaterialRepository = Depends(get_material_repo)
) -> RAGService:
    """Provide RAGService with its dependencies."""
    return RAGService(repo)


def get_llm_service() -> LLMService:
    """Provide LLMService (stateless, no DB dependency)."""
    return LLMService()