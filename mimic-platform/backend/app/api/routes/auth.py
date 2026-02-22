from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.auth import RegisterRequest, LoginRequest
from app.services.auth_service import AuthService
from app.db.session import get_db

router = APIRouter(prefix="")
service = AuthService()



@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    token = service.register(
        db,
        req.email,
        req.password,
        req.role
    )

    return {"token": token}


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    token = service.login(
        db,
        req.email,
        req.password
    )

    return {"token": token}