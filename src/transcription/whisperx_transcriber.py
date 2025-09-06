"""WhisperX-based transcription and diarization module"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch

from .whisper_transcriber import TranscriptSegment, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    def __init__(self, model_size: str = "base", device: str = "auto", hf_token: str = None):
        self.model_size = model_size
        self.hf_token = hf_token
        
        # Configure device for AMD GPU
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using AMD GPU for WhisperX: {device_name}")
            else:
                self.device = "cpu"
                logger.info("Using CPU for WhisperX")
        else:
            self.device = device

    def transcribe_and_diarize(self, audio_path: Path) -> Dict:
        """Full WhisperX pipeline: transcribe, align, and diarize"""
        logger.info(f"=== ENTERING WhisperX transcribe_and_diarize ===")
        logger.info(f"Audio path: {audio_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model size: {self.model_size}")
        
        try:
            logger.info("Importing whisper and whisperx...")
            import whisper
            import whisperx
            logger.info("Imports successful")
            
            logger.info(f"Starting WhisperX transcription and diarization: {audio_path}")
            
            # Step 1: Transcribe
            logger.info(f"Step 1: Transcribing with Whisper on {self.device}...")
            try:
                model = whisper.load_model(self.model_size, device=self.device)
                logger.info("Whisper model loaded successfully")
                result = model.transcribe(str(audio_path))
                logger.info(f"Transcription complete, detected language: {result['language']}")
                logger.info(f"Transcription result type: {type(result)}")
                logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                logger.info(f"Number of segments: {len(result['segments']) if isinstance(result, dict) and 'segments' in result else 'Unknown'}")
            except Exception as e:
                logger.error(f"Error in transcription step: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Step 2: Align
            try:
                logger.info(f"Step 2: Aligning segments on {self.device}...")
                language_code = result["language"]
                logger.info(f"Loading alignment model for language: {language_code}")
                model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
                logger.info(f"Running alignment...")
                aligned_result = whisperx.align(result["segments"], model_a, metadata, str(audio_path), self.device)
                logger.info(f"Alignment complete: {len(aligned_result['segments']) if isinstance(aligned_result, dict) and 'segments' in aligned_result else len(aligned_result)} aligned segments")
                logger.info(f"Aligned result type: {type(aligned_result)}")
                logger.info(f"Aligned result keys: {aligned_result.keys() if isinstance(aligned_result, dict) else 'Not a dict'}")
            except Exception as e:
                logger.error(f"Error in alignment step: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
            # Step 3: Diarize
            diarized_segments = []
            if self.hf_token:
                logger.info("Step 3: Running speaker diarization...")
                # Use current WhisperX 3.x API - import directly
                from whisperx.diarize import DiarizationPipeline
                diarization_pipeline = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
                
                logger.info("Running diarization pipeline...")
                diarization_result = diarization_pipeline(str(audio_path))
                logger.info(f"Diarization complete, result type: {type(diarization_result)}")
                
                # Assign speakers to segments
                logger.info("Assigning speakers to segments...")
                # assign_word_speakers returns the modified aligned_result directly
                speaker_assigned_result = whisperx.assign_word_speakers(
                    diarization_result, aligned_result
                )
                logger.info(f"Speaker assignment complete")
                
                result_segments = speaker_assigned_result["segments"]
                logger.info(f"Got {len(result_segments)} segments from speaker assignment")
                logger.info(f"First segment type: {type(result_segments[0]) if result_segments else 'None'}")
                if result_segments:
                    logger.info(f"First segment keys: {list(result_segments[0].keys()) if isinstance(result_segments[0], dict) else 'Not a dict'}")
                    logger.info(f"First segment: {result_segments[0]}")
                else:
                    logger.error("No segments returned from speaker assignment!")
                    logger.info("Using original aligned segments without speaker labels")
                    # Fallback to original segments if speaker assignment failed
                    original_segments = aligned_result["segments"] if isinstance(aligned_result, dict) else aligned_result
                    for seg in original_segments:
                        diarized_segments.append({
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "text": seg.get("text", ""),
                            "speaker": "SPEAKER_01"
                        })
                
                # Convert to our format - handle both dict and object formats
                for i, segment in enumerate(result_segments):
                    try:
                        if isinstance(segment, dict):
                            diarized_segments.append({
                                "start": segment["start"],
                                "end": segment["end"], 
                                "text": segment["text"],
                                "speaker": segment.get("speaker", "SPEAKER_UNKNOWN")
                            })
                        else:
                            # Handle object format
                            diarized_segments.append({
                                "start": getattr(segment, "start", 0),
                                "end": getattr(segment, "end", 0), 
                                "text": getattr(segment, "text", ""),
                                "speaker": getattr(segment, "speaker", "SPEAKER_UNKNOWN")
                            })
                    except Exception as e:
                        logger.error(f"Error processing segment {i}: {e}")
                        logger.error(f"Segment data: {segment}")
                        logger.error(f"Segment type: {type(segment)}")
                        raise
                
                # Merge consecutive segments from same speaker
                diarized_segments = self._merge_segments(diarized_segments)
                
            else:
                logger.warning("No HF token provided, skipping diarization")
                for segment in aligned_result["segments"]:
                    diarized_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speaker": "SPEAKER_01"
                    })
            
            # Create raw transcription result for compatibility
            raw_segments = []
            for seg in result["segments"]:
                raw_segments.append(TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"]
                ))
            
            raw_transcription = TranscriptionResult(
                segments=raw_segments,
                language=result["language"],
                full_text=result["text"],
                duration=raw_segments[-1].end if raw_segments else 0.0
            )
            
            logger.info(f"WhisperX processing complete: {len(diarized_segments)} diarized segments")
            
            return {
                "raw_transcription": raw_transcription,
                "diarized_segments": diarized_segments,
                "language": result["language"]
            }
            
        except Exception as e:
            logger.error(f"WhisperX transcription failed: {e}")
            raise

    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge consecutive segments from the same speaker"""
        if not segments:
            return segments
            
        merged = []
        current_segment = segments[0].copy()
        
        for segment in segments[1:]:
            if (current_segment['speaker'] == segment['speaker'] and 
                segment['start'] - current_segment['end'] < 2.0):  # 2 second gap tolerance
                # Merge with current segment
                current_segment['end'] = segment['end']
                current_segment['text'] += ' ' + segment['text']
            else:
                # Start new segment
                merged.append(current_segment)
                current_segment = segment.copy()
        
        merged.append(current_segment)
        return merged