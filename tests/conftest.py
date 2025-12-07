"""
pytest configuration and fixtures.

"""

import pytest
from unittest.mock import Mock, patch
from smart_qa.client import LLMClient


@pytest.fixture
def mock_api_key(monkeypatch):
    """
    Mock the API key environment variable.
    
    monkeypatch: pytest's tool for safely modifying environment
    
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key-for-testing")


@pytest.fixture
def mock_genai(mock_api_key):
    """
    Mock the Google Generative AI module.
    
    Returns a mock that pretends to be the real API.
    """
    with patch('smart_qa.client.genai') as mock:
        # Create a mock model
        mock_model = Mock()
        mock.GenerativeModel.return_value = mock_model
        
        # Mock the generate_content method
        mock_response = Mock()
        mock_response.text = "Mocked response"
        mock_model.generate_content.return_value = mock_response
        
        yield mock


@pytest.fixture
def client(mock_genai, tmp_path):
    """
    Create a test client with mocked API and temporary cache.
    
    tmp_path: pytest's built-in fixture for temporary directories
   
    """
    cache_dir = tmp_path / ".cache"
    return LLMClient(cache_dir=str(cache_dir))


@pytest.fixture
def sample_text():
    """
    Provide sample text for testing.
    
    """
    return """
    John Smith met with Jane Doe in New York on January 15, 2024.
    They discussed the merger between TechCorp and InnovateLabs.
    The meeting took place at the Empire State Building.
    """