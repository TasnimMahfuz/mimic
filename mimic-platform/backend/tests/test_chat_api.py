import pytest
from unittest.mock import patch, MagicMock


@patch('app.api.routes.chat.llm_service')
@patch('app.api.routes.chat.rag_service')
def test_chat_query_endpoint(mock_rag, mock_llm, client):
    """Test chat query endpoint with mocked services."""
    # Setup mocks
    mock_rag.retrieve_context.return_value = "Relevant context about the topic"
    mock_llm.generate_response.return_value = "Generated answer to the question"
    
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


@patch('app.api.routes.chat.rag_service')
def test_chat_query_no_context(mock_rag, client):
    """Test chat query when no context is found."""
    mock_rag.retrieve_context.return_value = "No relevant context found."
    
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


def test_chat_query_requires_auth(client):
    """Test that chat query requires authentication."""
    response = client.post(
        "/chat/query",
        json={"query": "test"}
    )
    
    assert response.status_code != 200