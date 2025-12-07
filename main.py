"""
Command-line interface for the Smart Q&A tool.

Usage examples:
    python main.py summarize --file document.txt
    python main.py ask --file context.txt --question "What is...?"
    python main.py extract --file report.txt --save entities.json
    python main.py --clear-cache
"""

import argparse
import sys
import json
from pathlib import Path
from smart_qa.client import LLMClient
from smart_qa.custom_exceptions import LLMAPIError


def read_file(file_path: str) -> str:
    """
    Read text from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File contents as string
        
    Why use Path?
    - Cross-platform (works on Windows, Mac, Linux)
    - Better error handling
    - Modern Python best practice
    """
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ Error: File '{file_path}' not found")
            sys.exit(1)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if not content.strip():
            print(f"❌ Error: File '{file_path}' is empty")
            sys.exit(1)
            
        return content
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)


def write_file(file_path: str, content: str) -> None:
    """
    Write content to a file.
    
    Args:
        file_path: Path to save the file
        content: Content to write
    """
    try:
        path = Path(file_path)
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Saved to: {file_path}")
    
    except Exception as e:
        print(f"❌ Error writing file: {e}")
        sys.exit(1)


def handle_summarize(args, client: LLMClient) -> None:
    """
    Handle the 'summarize' command.
    
    Flow:
    1. Read text (from file or stdin)
    2. Call client.summarize()
    3. Print or save result
    """
    print("📝 Summarizing text...\n")
    
    # Get input text
    if args.file:
        text = read_file(args.file)
    else:
        print("Enter text to summarize (press Ctrl+D or Ctrl+Z when done):")
        text = sys.stdin.read()
    
    try:
        # Call the API (or get from cache)
        summary = client.summarize(text)
        
        # Output
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(summary)
        print("="*60 + "\n")
        
        # Save if requested
        if args.save:
            write_file(args.save, summary)
    
    except LLMAPIError as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
        sys.exit(1)


def handle_ask(args, client: LLMClient) -> None:
    """
    Handle the 'ask' command.
    
    Requires:
    - Context (from --file or stdin)
    - Question (from --question)
    """
    print("❓ Answering question...\n")
    
    # Get context
    if args.file:
        context = read_file(args.file)
    else:
        print("Enter context (press Ctrl+D or Ctrl+Z when done):")
        context = sys.stdin.read()
    
    # Get question
    if not args.question:
        print("❌ Error: --question is required for 'ask' command")
        sys.exit(1)
    
    try:
        # Call the API
        answer = client.ask(context, args.question)
        
        # Output
        print("\n" + "="*60)
        print(f"QUESTION: {args.question}")
        print("="*60)
        print(answer)
        print("="*60 + "\n")
        
        # Save if requested
        if args.save:
            output = f"Question: {args.question}\n\nAnswer: {answer}"
            write_file(args.save, output)
    
    except LLMAPIError as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
        sys.exit(1)


def handle_extract(args, client: LLMClient) -> None:
    """
    Handle the 'extract' command.
    
    Extracts entities and returns structured JSON.
    """
    print("🔍 Extracting entities...\n")
    
    # Get input text
    if args.file:
        text = read_file(args.file)
    else:
        print("Enter text to extract entities from (press Ctrl+D or Ctrl+Z when done):")
        text = sys.stdin.read()
    
    try:
        # Call the API
        entities = client.extract_entities(text)
        
        # Pretty print JSON
        json_output = json.dumps(entities, indent=2)
        
        print("\n" + "="*60)
        print("EXTRACTED ENTITIES:")
        print("="*60)
        print(json_output)
        print("="*60 + "\n")
        
        # Save if requested
        if args.save:
            write_file(args.save, json_output)
    
    except LLMAPIError as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation Error: {e}")
        sys.exit(1)


def main():
    """
    Main entry point for the CLI.
    
    argparse basics:
    - Creates a parser
    - Defines subcommands (summarize, ask, extract)
    - Parses command-line arguments
    """
    
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Smart Q&A Tool - Summarize, Ask, and Extract from text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize a document
  python main.py summarize --file report.txt
  
  # Ask a question
  python main.py ask --file context.txt --question "What is the main point?"
  
  # Extract entities and save to file
  python main.py extract --file article.txt --save entities.json
  
  # Clear the cache
  python main.py --clear-cache
        """
    )
    
    # Global flags (work with any command)
    parser.add_argument(
        '--clear-cache',
        action='store_true',  # Boolean flag (no value needed)
        help='Clear the cache before running'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest='command',  # Stores which command was chosen
        help='Available commands'
    )
    
    # --- SUMMARIZE command ---
    summarize_parser = subparsers.add_parser(
        'summarize',
        help='Summarize text'
    )
    summarize_parser.add_argument(
        '--file',
        type=str,
        help='Path to text file to summarize'
    )
    summarize_parser.add_argument(
        '--save',
        type=str,
        help='Save summary to file'
    )
    
    # --- ASK command ---
    ask_parser = subparsers.add_parser(
        'ask',
        help='Ask a question about text'
    )
    ask_parser.add_argument(
        '--file',
        type=str,
        help='Path to context file'
    )
    ask_parser.add_argument(
        '--question',
        type=str,
        required=True,  # Must provide a question
        help='Question to ask'
    )
    ask_parser.add_argument(
        '--save',
        type=str,
        help='Save answer to file'
    )
    
    # --- EXTRACT command ---
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract entities from text'
    )
    extract_parser.add_argument(
        '--file',
        type=str,
        help='Path to text file'
    )
    extract_parser.add_argument(
        '--save',
        type=str,
        help='Save extracted entities to JSON file'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command and not args.clear_cache:
        parser.print_help()
        sys.exit(0)
    
    # Initialize client
    try:
        print("🚀 Initializing Smart Q&A Client...\n")
        client = LLMClient()
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        sys.exit(1)
    
    # Handle --clear-cache flag
    if args.clear_cache:
        print("🧹 Clearing cache...")
        client.clear_cache()
        print("✅ Cache cleared!\n")
        
        # If only clearing cache, exit here
        if not args.command:
            sys.exit(0)
    
    # Route to appropriate handler
    if args.command == 'summarize':
        handle_summarize(args, client)
    elif args.command == 'ask':
        handle_ask(args, client)
    elif args.command == 'extract':
        handle_extract(args, client)


if __name__ == '__main__':
    """
    This block only runs when the script is executed directly
    (not when imported as a module)
    
    Why?
    - Allows code reuse (can import functions without running main)
    - Standard Python practice
    """
    main()