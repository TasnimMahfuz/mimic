import pytest
from unittest.mock import patch, MagicMock
from app.services.rag_service import RAGService


@patch('app.services.rag_service.VectorStoreService')
@patch('app.services.rag_service.MaterialRepository')
def test_rag_ingest_with_vector_storage(mock_repo, mock_vector_store):
    """Test RAG service ingestion with vector storage."""
    # Mock repository
    mock_material = MagicMock()
    mock_material.id = 1
    mock_repo.return_value.save_material.return_value = mock_material
    
    # Mock vector store
    mock_vector_store.return_value.store_vectors.return_value = True
    
    service = RAGService()
    material_id = service.ingest(
        db=MagicMock(),
        title="Test Material",
        content="Test content with some text",
        user_id=1
    )
    
    assert material_id == 1
    mock_repo.return_value.save_material.assert_called_once()
    mock_repo.return_value.save_chunks.assert_called_once()
    mock_vector_store.return_value.store_vectors.assert_called_once()


@patch('app.services.rag_service.VectorStoreService')
@patch('app.services.rag_service.MaterialRepository')
def test_rag_retrieve_context(mock_repo, mock_vector_store):
    """Test retrieving context (Score strings are now removed by the service)."""
    mock_vector_store.return_value.retrieve_vectors.return_value = [
        {"text": "relevant chunk 1", "material_id": 1, "score": 0.95},
        {"text": "relevant chunk 2", "material_id": 1, "score": 0.87}
    ]
    
    service = RAGService()
    context = service.retrieve_context("what is this about?")
    
    # These should pass
    assert "relevant chunk 1" in context
    assert "relevant chunk 2" in context
    
    # REMOVED: assert "0.95" in context 
    # REMOVED: assert "0.87" in context
    # These fail now because the service only returns the text.