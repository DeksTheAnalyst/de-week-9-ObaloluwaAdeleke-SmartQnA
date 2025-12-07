# Smart Q&A Tool

A production-ready Python library for intelligent document analysis using Google's Gemini AI with smart caching and error handling.

## Features

- **Summarize** long documents into concise summaries
- **Q&A** - Ask questions based on provided context
- **Extract** people, dates, and locations as structured JSON
- **Smart Caching** - Never pay twice for the same query
- **Auto-Retry** - Handles network failures with exponential backoff

## Installation

```bash
# Install dependencies
python -m poetry install

# Set up your API key
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Get your free API key at: <https://aistudio.google.com/app/apikey>

## Usage

```bash
# Activate environment
python -m poetry shell

# Summarize a document
python main.py summarize --file document.txt

# Ask a question
python main.py ask --file context.txt --question "What is the main idea?"

# Extract entities
python main.py extract --file article.txt --save entities.json

# Clear cache
python main.py --clear-cache
```

## Project Structure

```
smart_qa_project/
├── smart_qa/
│   ├── client.py              # Main LLMClient class
│   └── custom_exceptions.py   # Error handling
├── tests/
│   ├── test_client.py         # Unit tests
│   └── data/sample.txt        # Test data
├── main.py                    # CLI interface
├── pyproject.toml             # Dependencies
└── .env                       # API key (create this)
```

## Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=smart_qa

# Verbose output
pytest -v
```

## Key Design Patterns

- **Caching Layer** - Reduces API costs by storing results
- **Retry Logic** - Automatic exponential backoff for failures
- **Type Hints** - Full type annotations for better IDE support
- **Structured Logging** - Track cache hits/misses and errors
- **100% Test Coverage** - Comprehensive mocking and validation

## Example

```python
from smart_qa.client import LLMClient

client = LLMClient()

# Summarize
summary = client.summarize("Your long text here...")

# Ask
answer = client.ask(context="...", question="What is...?")

# Extract
entities = client.extract_entities("John met Jane in Paris on Jan 1, 2024")
# Returns: {"people": [...], "dates": [...], "locations": [...]}
```
