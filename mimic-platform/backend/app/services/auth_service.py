from sqlalchemy.orm import Session
from starlette import status

from app.repositories.user_repo import UserRepository
from app.repositories.role_repo import RoleRepository
from app.core.security import (
    hash_password,
    verify_password,
    create_token
)


from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException


class AuthService:

    def register(self, db: Session, email, password, role):

        user_repo = UserRepository(db)
        role_repo = RoleRepository(db)

        db_role = role_repo.get_by_name(role)
        if not db_role:
            raise HTTPException(
            status_code=400,
            detail=f"Role '{role}' not found"
        )

        try:
            hashed = hash_password(password)
            user = user_repo.create(
                email=email,
                password_hash=hashed,
                role_id=db_role.id
            )
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        return create_token({
            "user_id": user.id,
            "role": db_role.name
        })


    def login(self, db: Session, email, password):

        user_repo = UserRepository(db)

        user = user_repo.get_by_email(email)

        if not user:
            raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid email or password"
        )

        if not verify_password(password, user.password_hash):
            raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid email or password"
        )

        return create_token({
            "user_id": user.id,
            "role": user.role.name
        })