import pytest
from unittest.mock import patch, MagicMock
from app.api.deps import get_rag_service, get_llm_service


@patch('app.services.rag_service.VectorStoreService')
@patch('app.services.rag_service.MaterialRepository')
def test_chat_query_endpoint(mock_repo_class, mock_vector_store_class, client):
    """Test chat query endpoint with mocked services."""
    # Setup mocks
    mock_repo = MagicMock()
    mock_repo_class.return_value = mock_repo
    
    mock_vector_store = MagicMock()
    mock_vector_store_class.return_value = mock_vector_store
    
    # Create real RAGService with mocked dependencies
    from app.services.rag_service import RAGService
    mock_rag_service = RAGService(mock_repo)
    mock_rag_service.retrieve_context = MagicMock(return_value="Relevant context about the topic")
    
    # Mock LLMService
    mock_llm_service = MagicMock()
    mock_llm_service.generate_response = MagicMock(return_value="Generated answer to the question")
    
    # Override dependencies
    client.app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    client.app.dependency_overrides[get_llm_service] = lambda: mock_llm_service
    
    # Register and login
    client.post("/auth/register", json={
        "email": "user@test.com",
        "password": "pass",
        "role": "student"
    })
    login_resp = client.post("/auth/login", json={
        "email": "user@test.com",
        "password": "pass"
    })
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Query chat endpoint
    response = client.post(
        "/chat/query",
        json={"query": "What is machine learning?"},
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "context" in data
    assert data["response"] == "Generated answer to the question"
    
    # Cleanup
    client.app.dependency_overrides.clear()


@patch('app.services.rag_service.VectorStoreService')
@patch('app.services.rag_service.MaterialRepository')
def test_chat_query_no_context(mock_repo_class, mock_vector_store_class, client):
    """Test chat query when no context is found."""
    mock_repo = MagicMock()
    mock_repo_class.return_value = mock_repo
    mock_vector_store = MagicMock()
    mock_vector_store_class.return_value = mock_vector_store
    
    from app.services.rag_service import RAGService
    mock_rag_service = RAGService(mock_repo)
    mock_rag_service.retrieve_context = MagicMock(return_value="No relevant context found.")
    
    client.app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    client.app.dependency_overrides[get_llm_service] = lambda: MagicMock()
    
    # Register and login
    client.post("/auth/register", json={
        "email": "user2@test.com",
        "password": "pass",
        "role": "student"
    })
    login_resp = client.post("/auth/login", json={
        "email": "user2@test.com",
        "password": "pass"
    })
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.post(
        "/chat/query",
        json={"query": "obscure topic xyz"},
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "Unable to find relevant context" in data["response"]
    
    client.app.dependency_overrides.clear()


def test_chat_query_requires_auth(client):
    """Test that chat query requires authentication."""
    response = client.post(
        "/chat/query",
        json={"query": "test"}
    )
    
    assert response.status_code != 200