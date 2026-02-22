from app.repositories.material_repo import MaterialRepository
from app.rag.loader import load_text
from app.rag.chunker import chunk_text


class RAGService:

    def __init__(self):
        self.repo = MaterialRepository()

    def ingest(self, db, title, content, user_id):

        text = load_text(content)

        material = self.repo.save_material(
            db,
            title,
            text,
            user_id
        )

        chunks = chunk_text(text)

        self.repo.save_chunks(
            db,
            material.id,
            chunks
        )

        return material.id