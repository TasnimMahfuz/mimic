from sqlalchemy.orm import Session
from app.models.user import User


class UserRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, email, password_hash, role_id):
        user = User(
            email=email,
            password_hash=password_hash,
            role_id=role_id
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_by_email(self, email):
        return self.db.query(User)\
            .filter(User.email == email)\
            .first()