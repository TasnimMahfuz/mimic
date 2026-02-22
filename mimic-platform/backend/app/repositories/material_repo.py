from app.models.course_material import CourseMaterial
from app.models.knowledge_chunk import KnowledgeChunk


class MaterialRepository:

    def save_material(self, db, title, content, user_id):

        material = CourseMaterial(
            title=title,
            content=content,
            uploaded_by=user_id
        )

        db.add(material)
        db.commit()
        db.refresh(material)

        return material

    def save_chunks(self, db, material_id, chunks):

        for c in chunks:
            db.add(
                KnowledgeChunk(
                    content=c,
                    material_id=material_id
                )
            )

        db.commit()