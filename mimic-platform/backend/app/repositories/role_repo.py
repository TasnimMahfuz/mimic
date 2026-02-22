from sqlalchemy.orm import Session
from app.models.role import Role


class RoleRepository:

    def __init__(self, db: Session):
        self.db = db

    def get_by_name(self, name: str):
        return self.db.query(Role)\
            .filter(Role.name == name)\
            .first()