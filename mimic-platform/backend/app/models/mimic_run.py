from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from datetime import datetime
from app.db.base import Base


class MIMICRun(Base):
    """
    Database model for MIMIC analysis runs.
    
    **Validates: Requirements 17.5, 17.6, 17.7**
    """
    __tablename__ = "mimic_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True, nullable=False)
    dataset_name = Column(String)
    result_path = Column(String)
    status = Column(String, default="processing")
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)