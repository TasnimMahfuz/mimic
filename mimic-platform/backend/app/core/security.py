from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

SECRET = "SUPER_SECRET"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["bcrypt"])


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)


def create_token(data: dict):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=6)
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)