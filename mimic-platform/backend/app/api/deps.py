from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import jwt
from app.core.security import SECRET, ALGORITHM

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