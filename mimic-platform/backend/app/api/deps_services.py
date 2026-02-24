from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.repositories.material_repo import MaterialRepository


def get_rag_service():
    repo = MaterialRepository()
    return RAGService(repo)


def get_llm_service():
    return LLMService()


#some depeendency inversion quick fix. 