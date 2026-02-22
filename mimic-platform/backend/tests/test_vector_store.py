import pytest
from unittest.mock import patch, MagicMock
from app.services.vector_store import VectorStoreService
from fastapi import HTTPException

# --- INITIALIZATION TESTS ---

@patch.dict('os.environ', {
    'PINECONE_API_KEY': 'test_pinecone_key',
    'HUGGINGFACE_API_KEY': 'test_hf_key',
    'PINECONE_HOST': 'https://test-host.pinecone.io'
})
def test_vector_store_initialization():
    """Test VectorStoreService initialization with all keys present."""
    service = VectorStoreService()
    assert service.pinecone_api_key == 'test_pinecone_key'
    assert service.huggingface_api_key == 'test_hf_key'
    assert "test-host" in service.pinecone_host

@patch('os.getenv')
def test_vector_store_missing_api_key(mock_getenv):
    """
    FIXED: Test that it fails if API key is missing.
    Patches os.getenv directly to bypass any local .env files.
    """
    # Simulate PINECONE_API_KEY being missing
    mock_getenv.side_effect = lambda key, default=None: None if key == 'PINECONE_API_KEY' else 'exists'
    
    with pytest.raises(ValueError, match="PINECONE_API_KEY not set"):
        VectorStoreService()

# --- LOGIC & API TESTS ---

@patch('requests.post')
@patch.dict('os.environ', {
    'PINECONE_API_KEY': 'test_key',
    'HUGGINGFACE_API_KEY': 'test_hf_key',
    'PINECONE_HOST': 'https://test-host.pinecone.io'
})
def test_get_embeddings_batching(mock_post):
    """
    FIXED: Test retrieving embeddings using the NEW batching logic.
    Expects a list-of-lists [[...]] from the HF API.
    """
    mock_response = MagicMock()
    # Mocking the HF return format: a list of embedding vectors
    mock_response.json.return_value = [[0.1, 0.2, 0.3]]
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    
    service = VectorStoreService()
    embeddings = service.get_embeddings(["test text"])
    
    # Assert that batching logic correctly handled the nested list
    assert len(embeddings) == 1 
    assert embeddings[0] == [0.1, 0.2, 0.3]
    assert isinstance(embeddings[0], list)

@patch('requests.post')
@patch.dict('os.environ', {
    'PINECONE_API_KEY': 'test_key',
    'HUGGINGFACE_API_KEY': 'test_hf_key',
    'PINECONE_HOST': 'https://test-host.pinecone.io'
})
def test_store_vectors(mock_post):
    """Test the full ingestion flow: Embeddings -> Format -> Pinecone Upsert."""
    # 1. Mock HuggingFace response (Batch of 2 vectors)
    hf_response = MagicMock()
    hf_response.status_code = 200
    hf_response.json.return_value = [[0.1], [0.2]]
    
    # 2. Mock Pinecone response
    pc_response = MagicMock()
    pc_response.status_code = 200
    pc_response.json.return_value = {"upserted_count": 2}
    
    # Logic: First call is HF, second is Pinecone
    mock_post.side_effect = [hf_response, pc_response]
    
    service = VectorStoreService()
    result = service.store_vectors(material_id=1, chunks=["chunk1", "chunk2"])
    
    assert result is True
    # Verify we actually attempted to hit Pinecone
    assert mock_post.call_count == 2