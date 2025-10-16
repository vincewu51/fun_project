#!/usr/bin/env python3
"""
Complete Audio Processing Pipeline

This script provides an end-to-end workflow for:
1. Transcribing audio files
2. Summarizing transcripts
3. Generating reports

Usage:
    python process_audio.py audio.mp3
    python process_audio.py audio.mp3 --model medium --summary-style meeting
    python process_audio.py ./audio_folder/ --batch
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import our modules
try:
    from transcribe import AudioTranscriber
    from summarize import TextSummarizer
except ImportError:
    print("Error: Cannot import transcribe or summarize modules.")
    print("Make sure you're running this from the audio_transcribe directory.")
    sys.exit(1)


class AudioProcessor:
    """Complete pipeline for audio transcription and summarization"""

    def __init__(
        self,
        whisper_model: str = "base",
        llm_model: str = "llama3.2",
        llm_backend: str = "ollama",
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the audio processor

        Args:
            whisper_model: Whisper model for transcription
            llm_model: LLM model for summarization
            llm_backend: LLM backend ('ollama' or 'llamacpp')
            device: Device for transcription ('cuda', 'cpu', or None)
            verbose: Print progress information
        """
        self.verbose = verbose

        # Initialize transcriber
        if self.verbose:
            print("=" * 60)
            print("Initializing Audio Processor")
            print("=" * 60)

        self.transcriber = AudioTranscriber(
            model_name=whisper_model,
            device=device
        )

        # Initialize summarizer
        try:
            self.summarizer = TextSummarizer(
                backend=llm_backend,
                model=llm_model
            )
        except ConnectionError as e:
            print(f"\nWarning: {e}")
            print("Summarization will be skipped.")
            self.summarizer = None

    def process_file(
        self,
        audio_path: Path,
        output_dir: Path,
        language: Optional[str] = None,
        summary_style: str = "detailed",
        skip_summary: bool = False
    ) -> dict:
        """
        Process a single audio file

        Args:
            audio_path: Path to audio file
            output_dir: Output directory
            language: Language code for transcription
            summary_style: Style for summary
            skip_summary: Skip summarization step

        Returns:
            Dictionary with processing results
        """
        results = {
            "file": audio_path.name,
            "success": False,
            "transcript_path": None,
            "summary_path": None,
            "error": None
        }

        try:
            # Step 1: Transcribe
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Processing: {audio_path.name}")
                print(f"{'=' * 60}\n")
                print("Step 1/2: Transcribing audio...")

            transcript_result = self.transcriber.transcribe(
                str(audio_path),
                language=language,
                verbose=self.verbose
            )

            # Save transcript
            transcript_dir = output_dir / "transcripts"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_dir / f"{audio_path.stem}.txt"

            self.transcriber.save_transcript(
                transcript_result,
                str(transcript_path),
                format="txt"
            )

            results["transcript_path"] = str(transcript_path)

            # Step 2: Summarize
            if not skip_summary and self.summarizer:
                if self.verbose:
                    print(f"\nStep 2/2: Generating summary...")

                summary_result = self.summarizer.summarize(
                    transcript_result['text'],
                    style=summary_style
                )

                # Save summary
                summary_dir = output_dir / "summaries"
                summary_dir.mkdir(parents=True, exist_ok=True)
                summary_path = summary_dir / f"{audio_path.stem}_summary.txt"

                self.summarizer.save_summary(
                    summary_result,
                    str(summary_path),
                    format="txt"
                )

                results["summary_path"] = str(summary_path)

            # Generate report
            if self.verbose:
                self._print_report(audio_path, transcript_result, results)

            results["success"] = True

        except Exception as e:
            results["error"] = str(e)
            if self.verbose:
                print(f"\nError processing {audio_path.name}: {e}")

        return results

    def process_batch(
        self,
        audio_dir: Path,
        output_dir: Path,
        language: Optional[str] = None,
        summary_style: str = "detailed",
        skip_summary: bool = False
    ) -> List[dict]:
        """
        Process multiple audio files

        Args:
            audio_dir: Directory containing audio files
            output_dir: Output directory
            language: Language code for transcription
            summary_style: Style for summary
            skip_summary: Skip summarization step

        Returns:
            List of processing results
        """
        # Find audio files
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
        audio_files = [
            f for f in audio_dir.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return []

        print(f"\nFound {len(audio_files)} audio file(s) to process\n")

        # Process each file
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            if self.verbose:
                print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

            result = self.process_file(
                audio_file,
                output_dir,
                language=language,
                summary_style=summary_style,
                skip_summary=skip_summary
            )
            results.append(result)

        # Print batch summary
        if self.verbose:
            self._print_batch_summary(results)

        return results

    def _print_report(self, audio_path: Path, transcript_result: dict, results: dict):
        """Print processing report for a single file"""
        print(f"\n{'=' * 60}")
        print(f"Processing Complete: {audio_path.name}")
        print(f"{'=' * 60}")
        print(f"Language: {transcript_result['metadata']['language']}")
        print(f"Duration: {transcript_result['segments'][-1]['end']:.2f}s" if transcript_result['segments'] else "N/A")
        print(f"Transcript: {results['transcript_path']}")
        if results['summary_path']:
            print(f"Summary: {results['summary_path']}")
        print(f"{'=' * 60}\n")

    def _print_batch_summary(self, results: List[dict]):
        """Print summary for batch processing"""
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        print(f"\n{'=' * 60}")
        print("Batch Processing Summary")
        print(f"{'=' * 60}")
        print(f"Total files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed files:")
            for result in results:
                if not result['success']:
                    print(f"  - {result['file']}: {result['error']}")

        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline for audio transcription and summarization"
    )

    # Input/output
    parser.add_argument(
        "input",
        help="Path to audio file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )

    # Transcription options
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=AudioTranscriber.AVAILABLE_MODELS,
        help="Whisper model for transcription (default: base)"
    )
    parser.add_argument(
        "-l", "--language",
        help="Source language code (e.g., 'en', 'es'). Auto-detect if not specified"
    )
    parser.add_argument(
        "-d", "--device",
        choices=["cuda", "cpu"],
        help="Device for transcription (default: auto-detect)"
    )

    # Summarization options
    parser.add_argument(
        "-s", "--summary-style",
        default="detailed",
        choices=list(TextSummarizer.SUMMARY_STYLES.keys()),
        help="Summary style (default: detailed)"
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2",
        help="LLM model for summarization (default: llama3.2)"
    )
    parser.add_argument(
        "--llm-backend",
        default="ollama",
        choices=["ollama", "llamacpp"],
        help="LLM backend (default: ollama)"
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip summarization step"
    )

    # Processing options
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all audio files in directory"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Initialize processor
    try:
        processor = AudioProcessor(
            whisper_model=args.model,
            llm_model=args.llm_model,
            llm_backend=args.llm_backend,
            device=args.device,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return 1

    # Prepare paths
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1

    # Process files
    try:
        if input_path.is_file():
            processor.process_file(
                input_path,
                output_dir,
                language=args.language,
                summary_style=args.summary_style,
                skip_summary=args.skip_summary
            )
        elif input_path.is_dir() and args.batch:
            processor.process_batch(
                input_path,
                output_dir,
                language=args.language,
                summary_style=args.summary_style,
                skip_summary=args.skip_summary
            )
        else:
            print(f"Error: {input_path} is a directory. Use --batch to process all files.")
            return 1

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
