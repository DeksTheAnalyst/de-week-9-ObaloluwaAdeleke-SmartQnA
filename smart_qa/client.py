import os
import json
import logging
import functools
import time
from pathlib import Path
from typing import Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
from .custom_exceptions import LLMAPIError

# Configure logging (helps you debug and track what's happening)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    """
    A professional LLM client with caching and error handling.
    
    """
    
    def __init__(self, cache_dir: str = ".cache") -> None:
        """Initialize the LLM client."""
        load_dotenv()
    
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please create a .env file with your API key."
            )
        
        
        genai.configure(api_key=api_key)
        
        # Try these models in order (v0.8.5 format)
        model_attempts = [
            'gemini-2.5-flash-latest',
            'gemini-1.5-pro-latest', 
            'gemini-pro',
            'gemini-1.5-flash',
            'gemini-1.5-pro'
        ]
        
        last_error = None
        for model_name in model_attempts:
            try:
                self.model = genai.GenerativeModel("gemini-2.5-flash")
                # Test it works
                logger.info(f"✓ Successfully loaded model: {"gemini-2.5-flash"}")
                break
            except Exception as e:
                last_error = e
                logger.debug(f"Failed to load {"gemini-2.5-flash"}: {e}")
                continue
        else:
            # No model worked
            raise ValueError(f"Could not initialize any model. Last error: {last_error}")
        
        # Set up caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "llm_cache.json"
        
        self._load_cache()
        
        logger.info("LLMClient initialized successfully")
    
    def _load_cache(self) -> None:
        """
        Load cache from disk (private method, indicated by leading underscore).

        """
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached entries")
            else:
                self.cache = {}
        except json.JSONDecodeError:
            logger.warning("Cache file corrupted, starting fresh")
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk (persists between program runs)."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _get_cache_key(self, method: str, *args) -> str:
        """
        Create a unique key for caching.
        
        """
        # Create a string representation of the arguments
        args_str = json.dumps(args, sort_keys=True)
        
        # Combine method name and arguments
        key_content = f"{method}:{args_str}"
        
        # Use hash for shorter keys (optional but cleaner)
        import hashlib
        key_hash = hashlib.md5(key_content.encode()).hexdigest()
        
        return f"{method}:{key_hash}"
    
    def _cached_call(self, method_name: str, func, *args):
        """
        Generic caching wrapper
        
        """
        cache_key = self._get_cache_key(method_name, *args)
        
        # Check cache first
        if cache_key in self.cache:
            logger.info(f"Cache HIT for {method_name}")
            return self.cache[cache_key]
        
        # Cache miss - call the actual function
        logger.info(f"Cache MISS for {method_name} - calling API")
        result = func(*args)
        
        # Save to cache
        self.cache[cache_key] = result
        self._save_cache()
        
        return result
    def _extract_text(response):
        """Safely extract text from Gemini responses."""
        # Newer Gemini responses
        if hasattr(response, "text") and response.text:
            return response.text

        # Older style
        try:
            return response.candidates[0].content.parts[0].text
        except:
            return None
    
    
    def _call_api_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    [prompt],  # request must be a list
                )

                # validate
                if not response or not hasattr(response, "text"):
                    raise LLMAPIError("Empty or invalid response from API")

                return response.text

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"API call failed (attempt {attempt+1}/{max_retries}). "
                        f"Retrying in {wait}s... Error: {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error("API call failed after retries.")
                    raise LLMAPIError(f"API call failed: {e}")


    
    def summarize(self, text: str) -> str:
        """
        Summarize the given text.
        
        Args:
            text: The text to summarize
            
        Returns:
            A concise summary
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Define the actual API call as a lambda
        api_call = lambda t: self._call_api_with_retry(
            f"Provide a concise summary of the following text:\n\n{t}"
        )
        
        # Use caching wrapper
        return self._cached_call("summarize", api_call, text)
    
    def ask(self, context: str, question: str) -> str:
        """
        Answer a question based ONLY on the provided context.
        
        Args:
            context: The context to search for answers
            question: The question to answer
            
        Returns:
            The answer based on the context
        """
        # Validate inputs
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Craft a prompt that restricts the model to the context
        prompt = f"""Based ONLY on the following context, answer the question.
If the answer is not in the context, say "I cannot answer based on the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        api_call = lambda p: self._call_api_with_retry(p)
        return self._cached_call("ask", api_call, prompt)


    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract structured entities from text.
        
        Returns:
            Dictionary with keys: "people", "dates", "locations"
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        prompt = f"""Extract the following entities from the text and return ONLY a JSON object:
- people: list of person names
- dates: list of dates mentioned
- locations: list of locations

Text:
{text}

Return ONLY valid JSON, no markdown formatting."""
        
        api_call = lambda t: self._call_api_with_retry(prompt)
        
        # Get result (possibly from cache)
        result = self._cached_call("extract_entities", api_call, text)
        
        # Parse JSON safely
        return self._parse_json_safely(result)
    
    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON, handling markdown code blocks
        
        """
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        if text.startswith("```"):
            text = text[3:]  # Remove ```
        if text.endswith("```"):
            text = text[:-3]  # Remove trailing ```
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMAPIError(f"Failed to parse JSON: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache = {}
        self._save_cache()
        logger.info("Cache cleared")