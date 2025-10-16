# Quick Start Guide

Get started with audio transcription and summarization in 5 minutes!

## Quick Command - Batch Process Everything

```bash
# Batch process all audio files in input/ folder with highest accuracy (Chinese auto-detected)
python process_audio.py input --batch --model large-v3 --language zh --summary-style meeting --device cuda
```

This command will:
- Process all audio files in the `input/` directory
- Use the largest, most accurate Whisper model (large-v3)
- Detect and transcribe Chinese language
- Generate meeting-style summaries
- Use GPU acceleration for faster processing
- Output transcripts to `output/transcripts/`
- Output summaries to `output/summaries/`

## Prerequisites Installation

### 1. Install FFmpeg
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 2. Install Ollama
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS/Windows: Download from https://ollama.com/download
```

### 3. Pull an LLM model
```bash
ollama pull llama3.2
```

### 4. Install Python dependencies
```bash
cd audio_transcribe
pip install -r requirements.txt
```

## Basic Usage Examples

### Example 1: Transcribe Audio
```bash
# Place your audio file in the input/ directory or use any path
python transcribe.py path/to/audio.mp3

# Output will be in: output/transcripts/audio.txt
```

### Example 2: Summarize Transcript
```bash
# Start Ollama if not running
ollama serve &

# Summarize the transcript
python summarize.py output/transcripts/audio.txt --style brief
```

### Example 3: Complete Pipeline
```bash
# Transcribe and summarize in one command
python process_audio.py interview.mp3 --model base --summary-style meeting
```

### Example 4: Batch Processing
```bash
# Put all your audio files in input/
cp *.mp3 input/

# Process all at once
python transcribe.py input/ --batch
python summarize.py output/transcripts/ --batch --style bullet
```

### Example 5: Create Subtitles
```bash
# Generate SRT subtitle file for a video
python transcribe.py video.mp4 --format srt --language en
```

## Common Options

### Transcription
- `--model tiny|base|small|medium|large` - Model size (larger = more accurate, slower)
- `--language en` - Specify language (or auto-detect)
- `--format txt|json|srt|vtt` - Output format
- `--device cuda|cpu` - Force GPU or CPU

### Summarization
- `--style brief|detailed|bullet|action|meeting` - Summary style
- `--model llama3.2|mistral|phi3` - Choose LLM model
- `--prompt "Your custom prompt"` - Custom summarization prompt

## Troubleshooting

**Problem**: `ffmpeg not found`
- **Solution**: Install FFmpeg (see Prerequisites)

**Problem**: `Cannot connect to ollama`
- **Solution**: Run `ollama serve` in another terminal

**Problem**: `CUDA out of memory`
- **Solution**: Use smaller Whisper model: `--model tiny` or `--model base`

**Problem**: Transcription is very slow
- **Solution**:
  - Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
  - Use smaller model for faster processing
  - Use GPU if available

## Next Steps

See [README.md](README.md) for:
- Complete documentation
- Advanced usage examples
- Python API usage
- Performance tuning tips
- Custom integration examples

## Quick Reference

```bash
# Basic transcription
python transcribe.py audio.mp3

# High-quality transcription
python transcribe.py audio.mp3 --model large-v3

# Fast transcription
python transcribe.py audio.mp3 --model tiny

# Meeting minutes
python process_audio.py meeting.mp3 --summary-style meeting

# Lecture notes
python process_audio.py lecture.m4a --summary-style bullet

# Translate to English
python transcribe.py audio.mp3 --task translate
```
