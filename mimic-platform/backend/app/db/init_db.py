from sqlalchemy.orm import Session
from app.models.role import Role


DEFAULT_ROLES = ["teacher", "student"]


def seed_roles(db: Session):

    for role_name in DEFAULT_ROLES:
        exists = (
            db.query(Role)
            .filter(Role.name == role_name)
            .first()
        )

        if not exists:
            db.add(Role(name=role_name))


def init_db(db: Session):
    seed_roles(db)
    db.commit()