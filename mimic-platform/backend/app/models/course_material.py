from sqlalchemy import Column, Integer, String, Text, ForeignKey
from app.db.base import Base


class CourseMaterial(Base):

    __tablename__ = "course_materials"

    id = Column(Integer, primary_key=True)

    title = Column(String)
    content = Column(Text)

    uploaded_by = Column(Integer, ForeignKey("users.id"))