#!/usr/bin/env python3
"""
Text Summarization using Local LLM

This script summarizes transcripts using a local LLM via llama.cpp or Ollama.
Supports multiple summarization styles and customizable prompts.
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import requests


class TextSummarizer:
    """Handle text summarization using local LLM"""

    SUMMARY_STYLES = {
        "brief": "Provide a brief 2-3 sentence summary of the key points.",
        "detailed": "Provide a detailed summary with main points, key insights, and important details.",
        "bullet": "Provide a bullet-point summary with key takeaways.",
        "action": "Summarize with focus on action items, decisions made, and next steps.",
        "meeting": "Summarize as meeting minutes including attendees, topics discussed, decisions, and action items."
    }

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the summarizer

        Args:
            backend: 'ollama' or 'llamacpp'
            model: Model name (e.g., 'llama3.2', 'mistral', 'phi3')
            base_url: Base URL for the API
        """
        self.backend = backend
        self.model = model
        self.base_url = base_url

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test if the LLM backend is accessible"""
        try:
            if self.backend == "ollama":
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                response.raise_for_status()
                print(f"Connected to Ollama at {self.base_url}")
            elif self.backend == "llamacpp":
                response = requests.get(f"{self.base_url}/health", timeout=5)
                response.raise_for_status()
                print(f"Connected to llama.cpp server at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to {self.backend} at {self.base_url}. "
                f"Make sure the server is running. Error: {e}"
            )

    def summarize(
        self,
        text: str,
        style: str = "detailed",
        custom_prompt: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> Dict:
        """
        Summarize text using local LLM

        Args:
            text: Text to summarize
            style: Summary style ('brief', 'detailed', 'bullet', 'action', 'meeting')
            custom_prompt: Custom prompt to override style
            max_length: Maximum length of summary (optional)

        Returns:
            Dictionary containing summary and metadata
        """
        # Build prompt
        if custom_prompt:
            prompt_instruction = custom_prompt
        else:
            prompt_instruction = self.SUMMARY_STYLES.get(
                style,
                self.SUMMARY_STYLES["detailed"]
            )

        prompt = f"""You are an expert at summarizing and analyzing text.

{prompt_instruction}

Text to summarize:
{text}

Summary:"""

        # Generate summary
        print(f"Generating {style} summary using {self.model}...")

        if self.backend == "ollama":
            summary_text = self._summarize_ollama(prompt, max_length)
        elif self.backend == "llamacpp":
            summary_text = self._summarize_llamacpp(prompt, max_length)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Prepare result
        result = {
            "summary": summary_text.strip(),
            "original_length": len(text),
            "summary_length": len(summary_text),
            "compression_ratio": len(text) / len(summary_text) if summary_text else 0,
            "metadata": {
                "model": self.model,
                "backend": self.backend,
                "style": style if not custom_prompt else "custom",
                "timestamp": datetime.now().isoformat()
            }
        }

        return result

    def _summarize_ollama(self, prompt: str, max_length: Optional[int]) -> str:
        """Generate summary using Ollama"""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {}
        }

        if max_length:
            payload["options"]["num_predict"] = max_length

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def _summarize_llamacpp(self, prompt: str, max_length: Optional[int]) -> str:
        """Generate summary using llama.cpp server"""
        url = f"{self.base_url}/completion"

        payload = {
            "prompt": prompt,
            "n_predict": max_length or 512,
            "temperature": 0.7,
            "stop": ["</s>", "User:", "Question:"],
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            return response.json()["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"llama.cpp API error: {e}")

    def summarize_from_file(
        self,
        file_path: str,
        style: str = "detailed",
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """
        Load and summarize text from file

        Args:
            file_path: Path to text file or JSON transcript
            style: Summary style
            custom_prompt: Custom prompt

        Returns:
            Summary result dictionary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file
        if file_path.suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = data.get('text', '')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        if not text.strip():
            raise ValueError(f"No text content found in {file_path}")

        # Summarize
        result = self.summarize(text, style=style, custom_prompt=custom_prompt)
        result["metadata"]["source_file"] = str(file_path)

        return result

    def save_summary(self, result: Dict, output_path: str, format: str = "txt"):
        """
        Save summary to file

        Args:
            result: Summary result dictionary
            output_path: Output file path
            format: Output format ('txt' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['summary'])
                f.write("\n\n")
                f.write(f"--- Metadata ---\n")
                f.write(f"Model: {result['metadata']['model']}\n")
                f.write(f"Style: {result['metadata']['style']}\n")
                f.write(f"Original length: {result['original_length']} chars\n")
                f.write(f"Summary length: {result['summary_length']} chars\n")
                f.write(f"Compression ratio: {result['compression_ratio']:.2f}x\n")
                f.write(f"Generated: {result['metadata']['timestamp']}\n")

        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize transcripts using local LLM"
    )
    parser.add_argument(
        "input_file",
        help="Path to transcript file (.txt or .json)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: auto-generated in ./output/summaries/)"
    )
    parser.add_argument(
        "-s", "--style",
        default="detailed",
        choices=list(TextSummarizer.SUMMARY_STYLES.keys()),
        help="Summary style (default: detailed)"
    )
    parser.add_argument(
        "-m", "--model",
        default="llama3.2",
        help="LLM model name (default: llama3.2)"
    )
    parser.add_argument(
        "-b", "--backend",
        default="ollama",
        choices=["ollama", "llamacpp"],
        help="LLM backend (default: ollama)"
    )
    parser.add_argument(
        "-u", "--url",
        default="http://localhost:11434",
        help="Base URL for LLM API (default: http://localhost:11434)"
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Custom prompt (overrides --style)"
    )
    parser.add_argument(
        "-f", "--format",
        default="txt",
        choices=["txt", "json"],
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum summary length in tokens"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in directory"
    )

    args = parser.parse_args()

    # Initialize summarizer
    try:
        summarizer = TextSummarizer(
            backend=args.backend,
            model=args.model,
            base_url=args.url
        )
    except ConnectionError as e:
        print(f"Error: {e}")
        print(f"\nTo start Ollama: ollama serve")
        print(f"To pull a model: ollama pull {args.model}")
        return

    # Get files to process
    input_path = Path(args.input_file)

    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir() and args.batch:
        input_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.json"))
        if not input_files:
            print(f"No text/json files found in {input_path}")
            return
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return

    # Set output directory
    if args.output and len(input_files) == 1:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "output" / "summaries"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir

    # Process files
    print(f"\nFound {len(input_files)} file(s) to summarize\n")

    for input_file in input_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {input_file.name}")
            print(f"{'='*60}\n")

            # Summarize
            result = summarizer.summarize_from_file(
                str(input_file),
                style=args.style,
                custom_prompt=args.prompt
            )

            # Determine output path
            if output_path.is_dir():
                output_file = output_path / f"{input_file.stem}_summary.{args.format}"
            else:
                output_file = output_path

            # Save summary
            summarizer.save_summary(result, str(output_file), format=args.format)

            print(f"\n{'='*60}")
            print(f"Summary Statistics:")
            print(f"Original: {result['original_length']} characters")
            print(f"Summary: {result['summary_length']} characters")
            print(f"Compression: {result['compression_ratio']:.2f}x")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue


if __name__ == "__main__":
    main()
