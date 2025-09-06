"""Whisper-based transcription module"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import subprocess
import tempfile
import shutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    segments: List[TranscriptSegment]
    language: str
    full_text: str
    duration: float


class WhisperTranscriber:
    def __init__(self, model_size: str = "base", language: Optional[str] = None, device: str = "auto"):
        self.model_size = model_size
        self.language = language
        self.device = device
        self._check_whisper_cpp()

    def _check_whisper_cpp(self):
        """Check if whisper.cpp is available"""
        # Skip binary check, use OpenAI Whisper instead
        raise RuntimeError("Using OpenAI Whisper instead of whisper.cpp binary")

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio file using whisper.cpp"""
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                output_file = temp_path / "transcript"
                
                cmd = [
                    "whisper",
                    str(audio_path),
                    "--model", self.model_size,
                    "--output-dir", str(temp_path),
                    "--output-file", "transcript",
                    "--output-json",
                    "--print-progress",
                ]
                
                # Add GPU flags for AMD ROCm if available
                if self.device == "auto" or self.device == "cuda":
                    cmd.extend(["--device", "cuda"])  # ROCm uses CUDA API
                
                if self.language:
                    cmd.extend(["--language", self.language])
                
                logger.info(f"Running whisper command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                json_file = temp_path / "transcript.json"
                if not json_file.exists():
                    raise RuntimeError(f"Whisper output file not found: {json_file}")
                
                with open(json_file, 'r') as f:
                    whisper_data = json.load(f)
                
                return self._parse_whisper_output(whisper_data)
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Whisper transcription failed: {e.stderr}")
            raise RuntimeError(f"Transcription failed: {e.stderr}")
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

    def _parse_whisper_output(self, whisper_data: Dict) -> TranscriptionResult:
        """Parse whisper.cpp JSON output"""
        segments = []
        
        for seg in whisper_data.get("transcription", []):
            segment = TranscriptSegment(
                start=seg.get("offsets", {}).get("from", 0) / 1000.0,  # Convert ms to seconds
                end=seg.get("offsets", {}).get("to", 0) / 1000.0,
                text=seg.get("text", "").strip()
            )
            segments.append(segment)
        
        full_text = " ".join(seg.text for seg in segments)
        duration = segments[-1].end if segments else 0.0
        language = whisper_data.get("result", {}).get("language", "unknown")
        
        return TranscriptionResult(
            segments=segments,
            language=language,
            full_text=full_text,
            duration=duration
        )


class OpenAIWhisperTranscriber:
    """Whisper transcriber using OpenAI's whisper package with AMD GPU support"""
    
    def __init__(self, model_size: str = "base", language: Optional[str] = None, device: str = "auto"):
        self.model_size = model_size
        self.language = language
        
        try:
            import whisper
            
            # Configure device for AMD GPU via ROCm
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"Using AMD GPU for transcription: {device_name}")
                else:
                    self.device = "cpu"
                    logger.info("Using CPU for transcription")
            else:
                self.device = device
            
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)
            
        except ImportError:
            raise RuntimeError("openai-whisper package not found. Install with: pip install openai-whisper")

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper with AMD GPU support"""
        try:
            logger.info(f"Transcribing with OpenAI Whisper on {self.device}: {audio_path}")
            
            result = self.model.transcribe(
                str(audio_path),
                language=self.language,
                verbose=True
            )
            
            segments = []
            for seg in result.get("segments", []):
                segment = TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip()
                )
                segments.append(segment)
            
            return TranscriptionResult(
                segments=segments,
                language=result.get("language", "unknown"),
                full_text=result.get("text", ""),
                duration=segments[-1].end if segments else 0.0
            )
            
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise


def get_transcriber(model_size: str = "base", language: Optional[str] = None, device: str = "auto"):
    """Get appropriate transcriber based on available tools"""
    try:
        return WhisperTranscriber(model_size, language, device)
    except RuntimeError:
        logger.info("whisper.cpp not found, using OpenAI Whisper with GPU support")
        return OpenAIWhisperTranscriber(model_size, language, device)