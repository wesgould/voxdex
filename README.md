# AI Auto Transcripts

An AI-powered podcast transcription system that provides high-quality speaker diarization using WhisperX, optional LLM speaker identification, and intelligent file management.

## Features

- **GPU-Accelerated Transcription**: Uses OpenAI Whisper with AMD GPU (ROCm) or NVIDIA GPU support
- **Advanced Speaker Diarization**: WhisperX integration with word-level alignment and pyannote.audio
- **Speaker Identification**: Optional LLM-based real name identification using OpenAI/Claude APIs
- **Multiple Output Formats**: Exports as TXT, JSON, and SRT subtitle files
- **Batch Processing**: Process multiple episodes from RSS feeds automatically
- **Three Output Versions**: Raw transcription, diarized (SPEAKER_01/02), and LLM-enhanced
- **Automated File Management**: Configurable retention policy for downloaded audio files
- **Storage Optimization**: Automatically cleans up old audio files while preserving transcripts

## How It Works

1. **Downloads audio** from podcast RSS feeds
2. **Transcribes** using Whisper (GPU-accelerated)
3. **Aligns** transcript with precise word-level timing using WhisperX
4. **Diarizes** to identify different speakers using pyannote.audio
5. **Merges** consecutive segments from same speaker
6. **Identifies** real speaker names using LLM (optional)
7. **Exports** three versions: raw, diarized, and enhanced

## Installation

### Prerequisites
- Python 3.10+
- AMD GPU with ROCm installed (or NVIDIA GPU with CUDA)
- HuggingFace account for speaker diarization
- Optional: OpenAI or Anthropic API key for speaker identification

### Setup

1. **Clone and setup virtual environment:**
   ```bash
   git clone <repository>
   cd ai-auto-transcripts
   chmod +x setup_venv.sh
   ./setup_venv.sh
   ```

2. **Activate environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Test GPU support:**
   ```bash
   python gpu_test.py
   ```

## Configuration

Edit `config.yaml` with your settings:

### Required Settings

```yaml
# Podcast feeds to process
feeds:
  - name: "Your Podcast Name"
    url: "https://example.com/podcast-feed.xml"
    max_episodes: 2  # Number of recent episodes to process
    enabled: true

# Speaker diarization (REQUIRED)
diarization:
  enabled: true
  hf_token: "hf_YOUR_TOKEN_HERE"  # HuggingFace token
  
# Output directory
output:
  base_dir: "/path/to/output/directory"  # Where transcripts are saved

# File retention (optional - for automatic cleanup)
file_retention:
  enabled: true
  retention_days: 7  # Keep audio files for 7 days after transcription
  audio_extensions: [".mp3", ".wav", ".m4a"]
  downloads_dir: "downloads"
```

### Optional Settings

```yaml
# Transcription settings
transcription:
  model_size: "base"  # tiny, base, small, medium, large
  language: null      # null for auto-detect, or "en", "es", etc.
  device: "auto"      # auto, cpu, cuda

# LLM speaker identification (optional)
llm:
  provider: "openai"  # openai or anthropic
  model: "gpt-4"
  # API key can be set here or in environment variables
  
# Diarization settings
diarization:
  min_speakers: 1
  max_speakers: 10
  model: "pyannote/speaker-diarization-3.1"
```

## API Keys Setup

### 1. HuggingFace Token (REQUIRED for diarization)

1. Create account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Create new token with "Read" permissions
4. Accept gated model access at https://hf.co/pyannote/speaker-diarization-3.1
5. Add token to `config.yaml` under `diarization.hf_token`

### 2. LLM API Keys (OPTIONAL - for speaker name identification)

**Option A: OpenAI**
1. Get API key from https://platform.openai.com/api-keys
2. Add to config or set environment variable `OPENAI_API_KEY`

**Option B: Anthropic Claude**
1. Get API key from https://console.anthropic.com
2. Add to config or set environment variable `ANTHROPIC_API_KEY`

**Environment Variables (Alternative):**
```bash
# Create .env file
cp .env.example .env
# Edit .env with your keys
```

## Usage

### Basic Usage
```bash
# Activate virtual environment
source venv/bin/activate

# Process all configured feeds
python main.py

# Process specific feed
python main.py --feed "https://example.com/feed.xml"

# Process single episode
python main.py --episode "https://example.com/episode.mp3"

# Custom output directory
python main.py --output-dir "/custom/path"

# File management commands
python main.py --file-stats      # Show download statistics
python main.py --cleanup-dry-run # Preview what would be deleted
python main.py --cleanup         # Actually delete old files
```

### Testing

```bash
# Test HuggingFace token access
python test_hf_token.py

# Test GPU detection
python gpu_test.py

# Test feed parsing (no audio download)
python test_pipeline.py
```

## Output Files

Files are organized in a hierarchical structure by podcast and episode:

```
output/
├── Security_Now/
│   └── SN_1041_Covering_All_the_Bases/
│       ├── SN_1041_raw.txt
│       ├── SN_1041_raw.json
│       ├── SN_1041_raw.srt
│       ├── SN_1041_diarized.txt
│       ├── SN_1041_diarized.json
│       ├── SN_1041_diarized.srt
│       ├── SN_1041_enhanced.txt
│       ├── SN_1041_enhanced.json
│       └── SN_1041_enhanced.srt
└── Other_Podcast/
    └── Episode_Directory/
        └── [Episode files...]

downloads/  # Audio files (automatically cleaned up based on retention policy)
├── Episode_1.mp3
├── Episode_2.wav
└── Episode_3.m4a
```

### File Organization Benefits
- **Organized by podcast**: Each podcast has its own directory
- **Episode-specific folders**: All files for an episode grouped together
- **Searchable filenames**: Easy to find with `grep` or `ripgrep` (e.g., `rg "SN_1041"` or `rg "_raw"`)
- **Clean structure**: No more long, cluttered filenames

### Three Output Versions Per Episode

#### 1. Raw Transcription (`*_raw.*`)
- Direct Whisper output
- Timestamps but no speaker labels
- Example: `SN_1041_raw.txt`

#### 2. Diarized Transcription (`*_diarized.*`)
- Generic speaker labels (SPEAKER_01, SPEAKER_02, etc.)
- Merged segments from same speaker
- Example: `SN_1041_diarized.srt`

#### 3. LLM-Enhanced Transcription (`*_enhanced.*`)
- Real speaker names identified by LLM (if API key provided)
- Otherwise same as diarized version
- Example: `SN_1041_enhanced.json`

Each version available in: `.txt`, `.json`, and `.srt` formats

## File Management

The system includes intelligent file management to optimize storage:

### Features:
- **Automatic Cleanup**: Runs after processing all feeds
- **Configurable Retention**: Keep files for any number of days (7 by default)
- **Safe Operation**: Never deletes transcripts, metadata, or output files
- **Transparent Logging**: Shows exactly what files are deleted/retained
- **Dry-Run Mode**: Preview changes before applying them

### Storage Management Commands:

```bash
# Check current storage usage
python main.py --file-stats
# Output: Total files: 15, Total size: 2.1 GB, Oldest: 2025-01-15, Newest: 2025-01-22

# See what would be cleaned up
python main.py --cleanup-dry-run
# Output: 8 files would be deleted, 7 files retained

# Perform actual cleanup
python main.py --cleanup
# Output: 8 files deleted, 7 files retained, 350MB freed
```

### Customizing Retention Policy:

```yaml
file_retention:
  enabled: true
  retention_days: 30        # Keep files for 30 days instead of 7
  audio_extensions: [".mp3", ".wav", ".m4a", ".flac"]  # Add FLAC support
  downloads_dir: "downloads"
```

## GPU Support

### AMD GPUs (ROCm)
- Requires ROCm installation on system
- Automatically detected with ROCm-compatible PyTorch
- Significant speedup for both transcription and diarization

### NVIDIA GPUs (CUDA)
- Requires CUDA installation
- Change PyTorch installation in `setup_venv.sh` if needed

### CPU Fallback
- System automatically falls back to CPU if no GPU detected
- Much slower but still functional

## Troubleshooting

### Common Issues

1. **"Could not download pyannote model"**
   - Ensure HF token has "Read" permissions
   - Accept terms at https://hf.co/pyannote/speaker-diarization-3.1
   - Test with `python test_hf_token.py`

2. **GPU not detected**
   - Check ROCm/CUDA installation
   - Verify with `python gpu_test.py`
   - Check PyTorch installation: `pip list | grep torch`

3. **"Feed parsing had issues"**
   - Warning only, RSS parsing still works
   - Safe to ignore UTF-8 encoding warnings

4. **Out of GPU memory**
   - Use smaller Whisper model: change `model_size` to "tiny" or "base"
   - Process fewer episodes: reduce `max_episodes` in config

5. **Disk space issues**
   - Check storage usage: `python main.py --file-stats`
   - Clean up old files: `python main.py --cleanup`
   - Adjust retention period in `config.yaml`

### Performance Tips

- **Faster processing**: Use "base" or "small" model instead of "large"
- **Better accuracy**: Use "medium" or "large" model (requires more GPU memory)
- **Batch processing**: Process multiple feeds overnight
- **Storage**: Each hour of audio ≈ 1-5MB of transcript files
- **File Management**: Set appropriate `retention_days` to balance storage and backup needs
- **Cleanup**: Use `--cleanup-dry-run` before making changes to retention settings

## Project Structure

```
ai-auto-transcripts/
├── main.py              # Main CLI entry point
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
├── setup_venv.sh        # Environment setup script
├── test_*.py           # Test scripts
└── src/
    ├── config/         # Configuration management
    ├── transcription/  # Whisper + WhisperX integration
    ├── diarization/    # Speaker separation (legacy)
    ├── llm/           # LLM speaker identification
    ├── export/        # Output file generation
    └── utils/         # RSS parsing, file management, and utilities
        ├── rss_parser.py    # RSS feed processing
        └── file_pruner.py   # File retention management
```

## Advanced Configuration

### Custom Models
```yaml
transcription:
  model_size: "large-v3"  # Use latest Whisper model

diarization:
  model: "pyannote/speaker-diarization-3.1"  # Alternative models
```

### Processing Limits
```yaml
feeds:
  - name: "Heavy Podcast"
    url: "https://example.com/feed.xml"
    max_episodes: 1        # Limit for large files
    enabled: true
```

### Output Customization
```yaml
output:
  formats: ["txt", "srt"]  # Skip JSON output
  include_timestamps: true # Include timing info

# File retention customization
file_retention:
  enabled: false          # Disable automatic cleanup
  retention_days: 14       # Keep files for 2 weeks
  downloads_dir: "audio"   # Custom download directory
```

## Support

For issues:
1. Check this README
2. Run test scripts to isolate problems
3. Check GPU/token access
4. Review log output for specific errors

## License

[License information here]