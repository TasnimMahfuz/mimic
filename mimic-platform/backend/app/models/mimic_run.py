from sqlalchemy import Column, Integer, String, ForeignKey
from app.db.base import Base


class MIMICRun(Base):
    __tablename__ = "mimic_runs"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String)
    result_path = Column(String)

    user_id = Column(Integer, ForeignKey("users.id"))