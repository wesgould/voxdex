"""Speaker diarization using pyannote.audio"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torchaudio

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str
    confidence: Optional[float] = None


@dataclass
class DiarizationResult:
    segments: List[SpeakerSegment]
    num_speakers: int
    duration: float


class SpeakerDiarizer:
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", 
                 min_speakers: int = 1, max_speakers: int = 10, hf_token: str = None):
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.hf_token = hf_token
        self.pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the diarization pipeline"""
        try:
            from pyannote.audio import Pipeline
            
            logger.info(f"Loading diarization model: {self.model_name}")
            if self.hf_token:
                self.pipeline = Pipeline.from_pretrained(self.model_name, use_auth_token=self.hf_token)
            else:
                self.pipeline = Pipeline.from_pretrained(self.model_name)
            
            # Check for AMD GPU support via ROCm (uses CUDA API)
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.pipeline = self.pipeline.to("cuda")
                logger.info(f"Using GPU for diarization: {device_name}")
            else:
                logger.info("Using CPU for diarization")
                
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {e}")
            raise RuntimeError(f"Diarization initialization failed: {e}")

    def diarize(self, audio_path: Path) -> DiarizationResult:
        """Perform speaker diarization on audio file"""
        try:
            logger.info(f"Running speaker diarization: {audio_path}")
            
            if not self.pipeline:
                raise RuntimeError("Diarization pipeline not initialized")
            
            diarization = self.pipeline(str(audio_path), 
                                      min_speakers=self.min_speakers,
                                      max_speakers=self.max_speakers)
            
            segments = []
            speakers = set()
            
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
                segments.append(SpeakerSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=speaker
                ))
            
            segments.sort(key=lambda x: x.start)
            
            duration = max(seg.end for seg in segments) if segments else 0.0
            
            result = DiarizationResult(
                segments=segments,
                num_speakers=len(speakers),
                duration=duration
            )
            
            logger.info(f"Diarization complete: {len(segments)} segments, {len(speakers)} speakers")
            return result
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def merge_with_transcription(self, transcription_segments: List, 
                                diarization_segments: List[SpeakerSegment]) -> List[Dict]:
        """Merge transcription segments with speaker labels"""
        merged_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.start
            trans_end = trans_seg.end
            trans_text = trans_seg.text
            
            speaker = self._find_dominant_speaker(trans_start, trans_end, diarization_segments)
            
            merged_segments.append({
                "start": trans_start,
                "end": trans_end,
                "text": trans_text,
                "speaker": speaker or "UNKNOWN"
            })
        
        return merged_segments

    def _find_dominant_speaker(self, start: float, end: float, 
                              diarization_segments: List[SpeakerSegment]) -> Optional[str]:
        """Find the speaker who speaks the most during a given time window"""
        speaker_durations = {}
        
        for dia_seg in diarization_segments:
            overlap_start = max(start, dia_seg.start)
            overlap_end = min(end, dia_seg.end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                speaker = dia_seg.speaker
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + overlap_duration
        
        if not speaker_durations:
            return None
        
        return max(speaker_durations.items(), key=lambda x: x[1])[0]


class SimpleDiarizer:
    """Fallback diarizer that assigns generic speaker labels based on silence detection"""
    
    def __init__(self, silence_threshold: float = 2.0):
        self.silence_threshold = silence_threshold

    def diarize(self, audio_path: Path) -> DiarizationResult:
        """Simple diarization based on silence gaps"""
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            duration = waveform.shape[1] / sample_rate
            
            segments = [SpeakerSegment(
                start=0.0,
                end=duration,
                speaker="SPEAKER_01"
            )]
            
            return DiarizationResult(
                segments=segments,
                num_speakers=1,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Simple diarization failed: {e}")
            raise

    def merge_with_transcription(self, transcription_segments: List, 
                                diarization_segments: List[SpeakerSegment]) -> List[Dict]:
        """Assign alternating speakers based on silence gaps"""
        merged_segments = []
        current_speaker_idx = 0
        speakers = ["SPEAKER_01", "SPEAKER_02"]
        last_end = 0.0
        
        for trans_seg in transcription_segments:
            if trans_seg.start - last_end > self.silence_threshold:
                current_speaker_idx = (current_speaker_idx + 1) % len(speakers)
            
            merged_segments.append({
                "start": trans_seg.start,
                "end": trans_seg.end,
                "text": trans_seg.text,
                "speaker": speakers[current_speaker_idx]
            })
            
            last_end = trans_seg.end
        
        return merged_segments


def get_diarizer(model_name: str = "pyannote/speaker-diarization-3.1", 
                min_speakers: int = 1, max_speakers: int = 10, hf_token: str = None) -> object:
    """Get appropriate diarizer based on available libraries"""
    try:
        return SpeakerDiarizer(model_name, min_speakers, max_speakers, hf_token)
    except RuntimeError:
        logger.warning("pyannote.audio not available, using simple diarization")
        return SimpleDiarizer()