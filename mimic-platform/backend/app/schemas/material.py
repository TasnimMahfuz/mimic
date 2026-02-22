from pydantic import BaseModel


class MaterialUpload(BaseModel):
    title: str
    content: str