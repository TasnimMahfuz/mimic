from sqlalchemy import Column, Integer, Text, ForeignKey
from app.db.base import Base


class KnowledgeChunk(Base):

    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True)

    content = Column(Text)

    material_id = Column(
        Integer,
        ForeignKey("course_materials.id")
    )