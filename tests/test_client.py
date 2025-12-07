"""
Comprehensive tests for LLMClient.

Test Categories:
1. Initialization tests
2. Caching tests
3. Error handling tests
4. API retry logic tests
5. Entity extraction tests
"""

import pytest
import json
from unittest.mock import Mock, patch, call
from smart_qa.client import LLMClient
from smart_qa.custom_exceptions import LLMAPIError


class TestInitialization:
    """Test client initialization."""
    
    def test_client_initializes_successfully(self, client):
        """Test that client initializes with mocked API."""
        assert client is not None
        assert client.model is not None
        assert client.cache == {}
    
    def test_client_fails_without_api_key(self, monkeypatch):
        """Test that client raises error when API key is missing."""
        # Remove the API key
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
            LLMClient()
    
    def test_cache_directory_created(self, client):
        """Test that cache directory is created."""
        assert client.cache_dir.exists()
        assert client.cache_dir.is_dir()


class TestCaching:
    """
    Test caching functionality.
    
    Critical requirement: API should only be called ONCE for identical requests
    """
    
    def test_summarize_caches_result(self, client, mock_genai):
        """
        Test that summarize caches results.
    
        """
        text = "This is a test document about caching."
        
        # Configure mock to return a specific response
        mock_model = mock_genai.GenerativeModel.return_value
        mock_model.generate_content.return_value.text = "Test summary"
        
        # First call - should hit API
        result1 = client.summarize(text)
        
        # Second call - should use cache
        result2 = client.summarize(text)
        
        # Verify results are identical
        assert result1 == result2
        assert result1 == "Test summary"
        
        # CRITICAL: API should only be called once
        assert mock_model.generate_content.call_count == 1
    
    def test_ask_caches_result(self, client, mock_genai):
        """Test that ask() caches results."""
        context = "The sky is blue."
        question = "What color is the sky?"
        
        mock_model = mock_genai.GenerativeModel.return_value
        mock_model.generate_content.return_value.text = "Blue"
        
        result1 = client.ask(context, question)
        result2 = client.ask(context, question)
        
        assert result1 == result2
        assert mock_model.generate_content.call_count == 1
    
    def test_different_inputs_not_cached(self, client, mock_genai):
        """Test that different inputs trigger new API calls."""
        mock_model = mock_genai.GenerativeModel.return_value
        mock_model.generate_content.return_value.text = "Response"
        
        client.summarize("First text")
        client.summarize("Second text")
        
        # Should call API twice (different inputs)
        assert mock_model.generate_content.call_count == 2
    
    def test_cache_persists_to_disk(self, client, tmp_path):
        """Test that cache is saved to disk."""
        mock_model = client.model
        mock_model.generate_content.return_value.text = "Cached result"
        
        client.summarize("Test")
        
        # Verify cache file was created
        cache_file = tmp_path / ".cache" / "llm_cache.json"
        assert cache_file.exists()
        
        # Verify cache content
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        assert len(cache_data) > 0
    
    def test_cache_loads_from_disk(self, tmp_path, mock_genai):
        """Test that cache is loaded from disk on init."""
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "llm_cache.json"
        
        # Pre-populate cache file
        existing_cache = {
            "test_key": "test_value"
        }
        with open(cache_file, 'w') as f:
            json.dump(existing_cache, f)
        
        # Create new client (should load existing cache)
        client = LLMClient(cache_dir=str(cache_dir))
        
        assert "test_key" in client.cache
        assert client.cache["test_key"] == "test_value"
    
    def test_clear_cache_removes_all_entries(self, client):
        """Test that clear_cache() removes all cached data."""
        # Add some cache entries
        client.cache["key1"] = "value1"
        client.cache["key2"] = "value2"
        client._save_cache()
        
        # Clear cache
        client.clear_cache()
        
        # Verify cache is empty
        assert client.cache == {}
        
        # Verify cache file reflects the change
        with open(client.cache_file, 'r') as f:
            cache_data = json.load(f)
        assert cache_data == {}


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_summarize_rejects_empty_text(self, client):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            client.summarize("")
        
        with pytest.raises(ValueError, match="Text cannot be empty"):
            client.summarize("   ")  # Only whitespace
    
    def test_ask_rejects_empty_context(self, client):
        """Test that empty context raises ValueError."""
        with pytest.raises(ValueError, match="Context cannot be empty"):
            client.ask("", "What is this?")
    
    def test_ask_rejects_empty_question(self, client):
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            client.ask("Some context", "")
    
    def test_api_error_raises_custom_exception(self, client):
        """Test that API errors are wrapped in LLMAPIError."""
        mock_model = client.model
        mock_model.generate_content.side_effect = Exception("API is down")
        
        with pytest.raises(LLMAPIError, match="API call failed"):
            client.summarize("Test text")


class TestRetryLogic:
    """Test automatic retry with exponential backoff."""
    
    def test_retries_on_failure(self, client):
        """
        Test that API calls are retried on failure.
        
        Scenario:
        - First 2 calls fail
        - Third call succeeds
        - Should return the successful result
        """
        mock_model = client.model
        
        # Fail twice, then succeed
        mock_model.generate_content.side_effect = [
            Exception("Temporary failure"),
            Exception("Temporary failure"),
            Mock(text="Success!")
        ]
        
        result = client.summarize("Test")
        
        # Should have retried and eventually succeeded
        assert result == "Success!"
        assert mock_model.generate_content.call_count == 3
    
    def test_gives_up_after_max_retries(self, client):
        """Test that retries stop after max attempts."""
        mock_model = client.model
        
        # Always fail
        mock_model.generate_content.side_effect = Exception("Persistent failure")
        
        with pytest.raises(LLMAPIError, match="API call failed"):
            client.summarize("Test")
        
        # Should have tried 3 times (default max_retries)
        assert mock_model.generate_content.call_count == 3


class TestEntityExtraction:
    """Test structured data extraction."""
    
    def test_extract_entities_returns_dict(self, client):
        """Test that extract_entities returns a dictionary."""
        mock_model = client.model
        mock_response = {
            "people": ["John Doe"],
            "dates": ["2024-01-01"],
            "locations": ["New York"]
        }
        mock_model.generate_content.return_value.text = json.dumps(mock_response)
        
        result = client.extract_entities("Test text")
        
        assert isinstance(result, dict)
        assert "people" in result
        assert "dates" in result
        assert "locations" in result
    
    def test_extract_entities_handles_markdown_json(self, client):
        """
        Test that markdown-formatted JSON is parsed correctly.
        
        LLMs often return: ```json {...} ```
        We need to handle this gracefully.
        """
        mock_model = client.model
        mock_response = {
            "people": ["Alice"],
            "dates": [],
            "locations": ["Paris"]
        }
        
        # Wrap in markdown code block
        markdown_json = f"```json\n{json.dumps(mock_response)}\n```"
        mock_model.generate_content.return_value.text = markdown_json
        
        result = client.extract_entities("Test text")
        
        assert result == mock_response
    
    def test_extract_entities_raises_on_invalid_json(self, client):
        """Test that invalid JSON raises LLMAPIError."""
        mock_model = client.model
        mock_model.generate_content.return_value.text = "This is not JSON"
        
        with pytest.raises(LLMAPIError, match="Failed to parse JSON"):
            client.extract_entities("Test text")
    
    def test_extract_entities_rejects_empty_text(self, client):
        """Test that empty input is rejected."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            client.extract_entities("")


class TestIntegrationWithTestData:
    """Integration tests using actual test data files."""
    
    def test_summarize_sample_file(self, client):
        """Test summarizing the sample data file."""
        from pathlib import Path
        
        # Load test data
        sample_file = Path(__file__).parent / "data" / "sample.txt"
        with open(sample_file, 'r') as f:
            text = f.read()
        
        mock_model = client.model
        mock_model.generate_content.return_value.text = "Renaissance summary"
        
        result = client.summarize(text)
        
        assert result == "Renaissance summary"
        assert mock_model.generate_content.called