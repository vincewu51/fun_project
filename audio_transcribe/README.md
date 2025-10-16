# Audio Transcription and Summarization

A complete pipeline for transcribing audio files and generating AI-powered summaries using local models. Built with OpenAI Whisper for transcription and local LLMs (Ollama/llama.cpp) for summarization.

## Features

- **Audio Transcription**: Convert speech to text using Whisper (tiny to large-v3 models)
- **Multiple Output Formats**: TXT, JSON, SRT, VTT subtitle formats
- **Local LLM Summarization**: Generate summaries using Ollama or llama.cpp
- **Multiple Summary Styles**: Brief, detailed, bullet points, action items, meeting minutes
- **Batch Processing**: Process multiple files at once
- **GPU Acceleration**: Automatic CUDA detection for faster processing
- **Language Support**: Auto-detect or specify 100+ languages
- **Privacy First**: Everything runs locally on your machine

## Prerequisites

### 1. FFmpeg (required for audio processing)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 2. Ollama (recommended for summarization)

**Install Ollama:**
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS/Windows
# Download from https://ollama.com/download
```

**Pull a model:**
```bash
ollama pull llama3.2         # Recommended: Fast and efficient
ollama pull mistral          # Alternative: Good for longer texts
ollama pull phi3             # Alternative: Lightweight option
```

**Start Ollama server:**
```bash
ollama serve
```

### 3. Python Environment

Python 3.8 or higher required.

## Installation

1. **Clone or navigate to the project:**
   ```bash
   cd audio_transcribe
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make scripts executable (Linux/macOS):**
   ```bash
   chmod +x transcribe.py summarize.py
   ```

## Quick Start

### Basic Usage

1. **Transcribe an audio file:**
   ```bash
   python transcribe.py path/to/audio.mp3
   ```

2. **Summarize the transcript:**
   ```bash
   python summarize.py output/transcripts/audio.txt
   ```

### One-Line Workflow

```bash
# Transcribe and summarize in sequence
python transcribe.py interview.mp3 -o output/transcripts/ && \
python summarize.py output/transcripts/interview.txt -s meeting
```

## Detailed Usage

### Transcription (`transcribe.py`)

**Basic transcription:**
```bash
python transcribe.py audio.mp3
```

**Advanced options:**
```bash
python transcribe.py audio.mp3 \
    --model medium \                    # Use medium-sized Whisper model
    --output ./my_transcripts/ \        # Custom output directory
    --format json \                     # Output as JSON
    --language en \                     # Specify English
    --task transcribe                   # Transcribe (vs translate)
```

**Available Whisper models:**
- `tiny` - Fastest, least accurate (~1GB VRAM)
- `base` - Good balance (default) (~1GB VRAM)
- `small` - Better accuracy (~2GB VRAM)
- `medium` - High accuracy (~5GB VRAM)
- `large` - Best accuracy (~10GB VRAM)
- `large-v2`, `large-v3` - Latest versions

**Batch processing:**
```bash
# Transcribe all audio files in a directory
python transcribe.py ./audio_folder/ --batch --format srt
```

**Supported audio formats:**
- MP3, WAV, M4A, FLAC, OGG, OPUS, WEBM

**Output formats:**
- `txt` - Plain text with metadata
- `json` - Full JSON with segments and timestamps
- `srt` - SubRip subtitle format
- `vtt` - WebVTT subtitle format

### Summarization (`summarize.py`)

**Basic summarization:**
```bash
python summarize.py transcript.txt
```

**Summary styles:**
```bash
# Brief summary (2-3 sentences)
python summarize.py transcript.txt --style brief

# Detailed summary
python summarize.py transcript.txt --style detailed

# Bullet points
python summarize.py transcript.txt --style bullet

# Action items and decisions
python summarize.py transcript.txt --style action

# Meeting minutes format
python summarize.py transcript.txt --style meeting
```

**Custom prompt:**
```bash
python summarize.py transcript.txt \
    --prompt "Extract all technical terms and explain them simply"
```

**Different models:**
```bash
python summarize.py transcript.txt \
    --model mistral \                   # Use Mistral instead of llama3.2
    --backend ollama \                  # Use Ollama (default)
    --url http://localhost:11434        # API endpoint
```

**Batch processing:**
```bash
# Summarize all transcripts in a directory
python summarize.py ./output/transcripts/ --batch --style bullet
```

## Project Structure

```
audio_transcribe/
├── transcribe.py           # Main transcription script
├── summarize.py            # Text summarization script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── input/                 # Place audio files here (optional)
├── output/
│   ├── transcripts/       # Generated transcripts
│   └── summaries/         # Generated summaries
└── models/                # Whisper model cache (auto-created)
```

## Examples

### Example 1: Interview Transcription

```bash
# Transcribe interview with high accuracy
python transcribe.py interview.mp3 \
    --model large-v3 \
    --format json \
    --output ./interviews/

# Generate meeting-style summary
python summarize.py ./interviews/interview.json \
    --style meeting \
    --output ./interviews/interview_summary.txt
```

### Example 2: Lecture Notes

```bash
# Transcribe lecture
python transcribe.py lecture.m4a --model medium

# Generate bullet-point summary
python summarize.py output/transcripts/lecture.txt \
    --style bullet \
    --output lecture_notes.txt
```

### Example 3: Batch Processing Multiple Files

```bash
# Transcribe all MP3 files in a folder
python transcribe.py ./recordings/ --batch --model base

# Summarize all transcripts
python summarize.py ./output/transcripts/ --batch --style detailed
```

### Example 4: Generate Subtitles

```bash
# Create SRT subtitles for a video
python transcribe.py video.mp4 --format srt --language en

# The output can be used with video players like VLC
```

## Configuration

### Using Different LLM Backends

**Ollama (default):**
```bash
python summarize.py transcript.txt --backend ollama --model llama3.2
```

**llama.cpp server:**
```bash
# Start llama.cpp server first
./llama-server -m model.gguf --port 8080

# Use with summarizer
python summarize.py transcript.txt \
    --backend llamacpp \
    --url http://localhost:8080
```

### GPU vs CPU

The transcription script automatically detects CUDA availability:
- **GPU available**: Uses CUDA for faster processing
- **No GPU**: Falls back to CPU (slower but works)

Force CPU usage:
```bash
python transcribe.py audio.mp3 --device cpu
```

## Performance Tips

1. **Model selection**:
   - For speed: Use `tiny` or `base`
   - For accuracy: Use `medium` or `large`
   - For batch processing: Start with `base`, upgrade if needed

2. **GPU acceleration**:
   - Transcription is 5-10x faster on GPU
   - Ensure CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Batch processing**:
   - Process multiple files at once to save time
   - Use lower quality models for quick drafts

4. **Summary length**:
   - Use `--max-length` to control token generation
   - Longer transcripts may need larger LLM models

## Troubleshooting

### Transcription Issues

**Error: "ffmpeg not found"**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu/Debian
brew install ffmpeg      # macOS
```

**Error: "CUDA out of memory"**
- Use a smaller Whisper model: `--model base` or `--model tiny`
- Force CPU: `--device cpu`

**Poor transcription quality:**
- Try a larger model: `--model medium` or `--model large`
- Specify the language: `--language en`
- Ensure audio quality is good (clear speech, minimal background noise)

### Summarization Issues

**Error: "Cannot connect to ollama"**
```bash
# Start Ollama server
ollama serve

# Pull the model you're trying to use
ollama pull llama3.2
```

**Error: "Model not found"**
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.2
```

**Summary is too long/short:**
- Use `--max-length` to control length
- Try different `--style` options
- Use custom `--prompt` for specific needs

**Slow summarization:**
- Use smaller models: `phi3` or `tinyllama`
- Consider using llama.cpp with quantized models for speed

## Advanced Usage

### Custom Python Integration

**Transcription in Python:**
```python
from transcribe import AudioTranscriber

transcriber = AudioTranscriber(model_name="base", device="cuda")
result = transcriber.transcribe("audio.mp3", language="en")
transcriber.save_transcript(result, "output.txt", format="txt")
```

**Summarization in Python:**
```python
from summarize import TextSummarizer

summarizer = TextSummarizer(backend="ollama", model="llama3.2")
result = summarizer.summarize_from_file("transcript.txt", style="bullet")
summarizer.save_summary(result, "summary.txt", format="txt")
```

### Pipeline Automation

Create a bash script for automated processing:

```bash
#!/bin/bash
# process_audio.sh

INPUT_DIR="./input"
TRANSCRIPTS="./output/transcripts"
SUMMARIES="./output/summaries"

# Transcribe all audio files
python transcribe.py "$INPUT_DIR" --batch --model base

# Summarize all transcripts
python summarize.py "$TRANSCRIPTS" --batch --style meeting

echo "Processing complete!"
```

## System Requirements

**Minimum:**
- CPU: Modern multi-core processor
- RAM: 8GB (for base Whisper model)
- Disk: 5GB free space
- OS: Linux, macOS, or Windows

**Recommended:**
- GPU: NVIDIA GPU with 8GB+ VRAM (for large models)
- RAM: 16GB
- Disk: 20GB free space (for model caching)

## Model Download Sizes

**Whisper models:**
- tiny: ~75 MB
- base: ~145 MB
- small: ~488 MB
- medium: ~1.5 GB
- large: ~3 GB

**LLM models (Ollama):**
- llama3.2: ~2 GB
- mistral: ~4 GB
- phi3: ~2.3 GB

Models are cached after first download.

## Privacy and Security

- All processing happens locally on your machine
- No data is sent to external servers
- Audio files and transcripts remain on your system
- Perfect for sensitive content (interviews, meetings, medical records)

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## License

Not specified. See main repository for licensing information.

## Useful Links

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Ollama](https://ollama.com/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [FFmpeg](https://ffmpeg.org/)

## Acknowledgments

Built with:
- OpenAI Whisper for speech recognition
- Ollama/llama.cpp for local LLM inference
- PyTorch for deep learning
