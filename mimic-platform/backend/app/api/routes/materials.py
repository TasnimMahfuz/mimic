from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session

from app.schemas.material import MaterialUpload
from app.services.rag_service import RAGService
from app.api.deps import require_teacher
from app.db.session import SessionLocal


router = APIRouter(prefix="")

rag_service = RAGService()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@router.post("/upload")
def upload_material(
    file: UploadFile = File(...),
    user=Depends(require_teacher),
    db: Session = Depends(get_db)
):

    content = file.file.read().decode()

    material_id = rag_service.ingest(
        db,
        file.filename,
        content,
        user["user_id"]
    )

    return {"material_id": material_id}