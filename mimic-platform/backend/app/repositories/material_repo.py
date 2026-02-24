from sqlalchemy.orm import Session
from app.models.course_material import CourseMaterial
from app.models.knowledge_chunk import KnowledgeChunk


class MaterialRepository:

    def save_material(
        self, 
        db: Session, 
        title: str, 
        content: str, 
        user_id: int,
        vector_status: str = "pending"
    ) -> CourseMaterial:
        """Save material with vector status tracking."""
        material = CourseMaterial(
            title=title,
            content=content,
            uploaded_by=user_id,
            vector_status=vector_status
        )

        db.add(material)
        db.commit()
        db.refresh(material)

        return material

    def save_chunks(self, db: Session, material_id: int, chunks: list):
        """Save knowledge chunks to database."""
        for c in chunks:
            db.add(
                KnowledgeChunk(
                    content=c,
                    material_id=material_id
                )
            )

        db.commit()

    def update_material_vector_status(
        self,
        db: Session,
        material_id: int,
        status: str
    ):
        """Update vector sync status for a material."""
        material = db.query(CourseMaterial).filter(
            CourseMaterial.id == material_id
        ).first()
        
        if material:
            material.vector_status = status
            db.commit()