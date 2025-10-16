#!/usr/bin/env python3
"""
Audio Transcription using OpenAI Whisper (local model)

This script transcribes audio files using the Whisper model running locally.
Supports multiple audio formats: mp3, wav, m4a, flac, ogg, etc.
"""

import argparse
import os
import json
import sys
import time
import threading
from pathlib import Path
from datetime import datetime
import torch
import whisper
from typing import Dict, Optional


class AudioTranscriber:
    """Handle audio transcription using Whisper model"""

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        Initialize the transcriber with a Whisper model

        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model must be one of {self.AVAILABLE_MODELS}")

        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"Model loaded successfully!")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True
    ) -> Dict:
        """
        Transcribe an audio file

        Args:
            audio_path: Path to the audio file
            language: Source language code (e.g., 'en', 'es', 'fr'). None for auto-detect
            task: 'transcribe' or 'translate' (translate to English)
            verbose: Print progress information

        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if verbose:
            print(f"\nTranscribing: {audio_path}")
            print(f"Task: {task}, Language: {language or 'auto-detect'}")

        # Load audio to get duration
        audio = whisper.load_audio(audio_path)
        duration = len(audio) / whisper.audio.SAMPLE_RATE

        if verbose:
            print(f"Audio duration: {duration:.1f}s")

        # Progress indicator
        stop_spinner = threading.Event()
        elapsed_time = [0]  # Use list to allow modification in nested function

        def spinner():
            """Animated progress indicator"""
            chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            idx = 0
            start_time = time.time()
            while not stop_spinner.is_set():
                elapsed_time[0] = time.time() - start_time
                est_progress = min(int((elapsed_time[0] / (duration * 0.5)) * 100), 99)  # Rough estimate
                if verbose:
                    sys.stdout.write(f'\r{chars[idx]} Transcribing... {est_progress}% (Elapsed: {elapsed_time[0]:.1f}s)')
                    sys.stdout.flush()
                idx = (idx + 1) % len(chars)
                time.sleep(0.1)

        # Start spinner if verbose
        if verbose:
            spinner_thread = threading.Thread(target=spinner, daemon=True)
            spinner_thread.start()

        # Transcribe
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False
        )

        # Stop spinner
        if verbose:
            stop_spinner.set()
            spinner_thread.join(timeout=0.5)
            # Clear line and show completion
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            if result.get('segments'):
                total_transcribed = result['segments'][-1]['end']
                print(f"✓ Transcription complete! ({total_transcribed:.1f}s transcribed in {elapsed_time[0]:.1f}s)")
            else:
                print(f"✓ Transcription complete! ({elapsed_time[0]:.1f}s)")

        # Add metadata
        result['metadata'] = {
            'file': os.path.basename(audio_path),
            'model': self.model_name,
            'device': self.device,
            'timestamp': datetime.now().isoformat(),
            'language': result.get('language', 'unknown'),
            'task': task,
            'duration': f"{duration:.2f}s"
        }

        return result

    def save_transcript(self, result: Dict, output_path: str, format: str = "txt"):
        """
        Save transcription to file

        Args:
            result: Transcription result dictionary
            output_path: Path to save the transcript
            format: Output format ('txt', 'json', 'srt', 'vtt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['text'].strip())
                f.write("\n\n")
                f.write(f"--- Metadata ---\n")
                for key, value in result['metadata'].items():
                    f.write(f"{key}: {value}\n")

        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        elif format == "srt":
            self._save_srt(result, output_path)

        elif format == "vtt":
            self._save_vtt(result, output_path)

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Transcript saved to: {output_path}")

    def _save_srt(self, result: Dict, output_path: Path):
        """Save transcription in SRT subtitle format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start = self._format_timestamp(segment['start'], srt=True)
                end = self._format_timestamp(segment['end'], srt=True)
                text = segment['text'].strip()

                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")

    def _save_vtt(self, result: Dict, output_path: Path):
        """Save transcription in WebVTT subtitle format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for segment in result['segments']:
                start = self._format_timestamp(segment['start'])
                end = self._format_timestamp(segment['end'])
                text = segment['text'].strip()

                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")

    @staticmethod
    def _format_timestamp(seconds: float, srt: bool = False) -> str:
        """Format seconds as timestamp string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        if srt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper (local)"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file or directory containing audio files"
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=AudioTranscriber.AVAILABLE_MODELS,
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: ./output/transcripts/)"
    )
    parser.add_argument(
        "-f", "--format",
        default="txt",
        choices=["txt", "json", "srt", "vtt"],
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "-l", "--language",
        help="Source language code (e.g., 'en', 'es', 'zh'). Auto-detect if not specified"
    )
    parser.add_argument(
        "-t", "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: 'transcribe' or 'translate' to English (default: transcribe)"
    )
    parser.add_argument(
        "-d", "--device",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all audio files in directory"
    )

    args = parser.parse_args()

    # Initialize transcriber
    transcriber = AudioTranscriber(model_name=args.model, device=args.device)

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "output" / "transcripts"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get audio files to process
    audio_path = Path(args.audio_file)
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm'}

    if audio_path.is_file():
        audio_files = [audio_path]
    elif audio_path.is_dir() and args.batch:
        audio_files = [f for f in audio_path.iterdir() if f.suffix.lower() in audio_extensions]
        if not audio_files:
            print(f"No audio files found in {audio_path}")
            return
    else:
        print(f"Error: {audio_path} is not a valid file or directory")
        return

    # Process each audio file
    print(f"\nFound {len(audio_files)} audio file(s) to process\n")

    for audio_file in audio_files:
        try:
            # Transcribe
            result = transcriber.transcribe(
                str(audio_file),
                language=args.language,
                task=args.task,
                verbose=True
            )

            # Generate output filename
            output_name = audio_file.stem + f".{args.format}"
            output_path = output_dir / output_name

            # Save transcript
            transcriber.save_transcript(result, str(output_path), format=args.format)

            print(f"\n{'='*60}")
            print(f"Transcription complete!")
            print(f"Language: {result['metadata']['language']}")
            print(f"Duration: {result['segments'][-1]['end']:.2f} seconds" if result['segments'] else "N/A")
            print(f"Output: {output_path}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue


if __name__ == "__main__":
    main()
