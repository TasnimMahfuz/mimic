import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_service import LLMService


@patch.dict('os.environ', {
    'GROQ_API_KEY': 'test_groq_key'
})
def test_llm_service_initialization():
    """Test that LLMService initializes with Groq API key."""
    service = LLMService()
    assert service.client is not None


@patch.dict('os.environ', {
    'GROQ_API_KEY': ''
})
def test_llm_service_missing_api_key():
    """Test that LLMService raises error without API key."""
    with pytest.raises(ValueError, match="GROQ_API_KEY not set"):
        LLMService()


@patch('app.services.llm_service.ChatGroq')
def test_generate_response(mock_groq):
    """Test response generation with mocked Groq API."""
    mock_response = MagicMock()
    mock_response.content = "This is the generated answer."
    mock_groq.return_value.invoke.return_value = mock_response
    
    with patch.dict('os.environ', {'GROQ_API_KEY': 'test_key'}):
        service = LLMService()
        service.client = MagicMock()
        service.client.invoke.return_value = mock_response
        
        result = service.generate_response(
            context="Test context about astronomy",
            query="What is a star?"
        )
        
        assert result == "This is the generated answer."
        service.client.invoke.assert_called_once()


@patch('app.services.llm_service.ChatGroq')
def test_generate_response_with_fallback(mock_groq):
    """Test response generation fallback on error."""
    with patch.dict('os.environ', {'GROQ_API_KEY': 'test_key'}):
        service = LLMService()
        service.client = MagicMock()
        service.client.invoke.side_effect = Exception("API Error")
        
        result = service.generate_response_with_fallback(
            context="Test context",
            query="Test query"
        )
        
        assert "Error generating response" in result