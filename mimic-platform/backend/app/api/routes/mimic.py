from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import SessionLocal

from app.services.mimic_service import MIMICService

router = APIRouter()
service = MIMICService()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/run")
def run_mimic(data: dict, db: Session = Depends(get_db)):
    return service.run(
        db,
        data["user_id"],
        data["dataset"]
    )