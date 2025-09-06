"""Export transcripts in various formats"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TranscriptExporter:
    def __init__(self, output_dir: str = "outputs", include_timestamps: bool = True):
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.include_timestamps = include_timestamps

    def export_all_formats(self, episode, raw_transcription, 
                          diarized_segments: List[Dict], llm_segments: List[Dict],
                          speaker_mappings: Dict[str, str] = None,
                          processing_metadata: Dict = None) -> Dict[str, Path]:
        """Export all three transcript formats plus metadata"""
        # Create episode-specific output directory
        safe_title = self._sanitize_filename(episode.title)
        safe_podcast_name = self._sanitize_filename(episode.podcast_name or "Unknown_Podcast")
        
        episode_dir = self.base_output_dir / safe_podcast_name / safe_title
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Use episode identifier for filenames, fallback to safe title
        filename_prefix = episode.episode_identifier or safe_title
        
        outputs = {}
        
        outputs["raw"] = self.export_raw_transcript(filename_prefix, raw_transcription, episode_dir)
        outputs["diarized"] = self.export_diarized_transcript(filename_prefix, diarized_segments, episode_dir)
        outputs["llm_enhanced"] = self.export_llm_transcript(filename_prefix, llm_segments, speaker_mappings, episode_dir)
        outputs["metadata"] = self.export_metadata(filename_prefix, episode, processing_metadata, episode_dir)
        
        return outputs

    def export_raw_transcript(self, filename_prefix: str, transcription_result, episode_dir: Path) -> Path:
        """Export raw Whisper transcription with timestamps"""
        outputs = {}
        
        txt_content = self._format_raw_text(transcription_result)
        outputs["txt"] = self._write_file(f"{filename_prefix}_raw.txt", txt_content, episode_dir)
        
        json_content = self._format_raw_json(transcription_result)
        outputs["json"] = self._write_file(f"{filename_prefix}_raw.json", json.dumps(json_content, indent=2), episode_dir)
        
        srt_content = self._format_raw_srt(transcription_result)
        outputs["srt"] = self._write_file(f"{filename_prefix}_raw.srt", srt_content, episode_dir)
        
        return outputs["txt"]

    def export_diarized_transcript(self, filename_prefix: str, segments: List[Dict], episode_dir: Path) -> Path:
        """Export diarized transcript with speaker labels"""
        txt_content = self._format_diarized_text(segments)
        self._write_file(f"{filename_prefix}_diarized.txt", txt_content, episode_dir)
        
        json_content = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "type": "diarized_transcript",
                "num_segments": len(segments)
            },
            "segments": segments
        }
        self._write_file(f"{filename_prefix}_diarized.json", json.dumps(json_content, indent=2), episode_dir)
        
        srt_content = self._format_segments_srt(segments)
        srt_file = self._write_file(f"{filename_prefix}_diarized.srt", srt_content, episode_dir)
        
        return srt_file

    def export_llm_transcript(self, filename_prefix: str, segments: List[Dict], 
                             speaker_mappings: Dict[str, str] = None, episode_dir: Path = None) -> Path:
        """Export LLM-enhanced transcript with identified speakers"""
        txt_content = self._format_diarized_text(segments)
        self._write_file(f"{filename_prefix}_enhanced.txt", txt_content, episode_dir)
        
        json_content = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "type": "llm_enhanced_transcript",
                "num_segments": len(segments),
                "speaker_mappings": speaker_mappings or {}
            },
            "segments": segments
        }
        self._write_file(f"{filename_prefix}_enhanced.json", json.dumps(json_content, indent=2), episode_dir)
        
        srt_content = self._format_segments_srt(segments)
        srt_file = self._write_file(f"{filename_prefix}_enhanced.srt", srt_content, episode_dir)
        
        return srt_file

    def _format_raw_text(self, transcription_result) -> str:
        """Format raw transcription as plain text"""
        lines = []
        
        for segment in transcription_result.segments:
            if self.include_timestamps:
                timestamp = self._format_timestamp(segment.start)
                lines.append(f"[{timestamp}] {segment.text}")
            else:
                lines.append(segment.text)
        
        return "\n".join(lines)

    def _format_raw_json(self, transcription_result) -> Dict:
        """Format raw transcription as JSON"""
        return {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "type": "raw_transcript",
                "language": transcription_result.language,
                "duration": transcription_result.duration,
                "num_segments": len(transcription_result.segments)
            },
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "confidence": seg.confidence
                }
                for seg in transcription_result.segments
            ]
        }

    def _format_raw_srt(self, transcription_result) -> str:
        """Format raw transcription as SRT subtitle file"""
        lines = []
        
        for i, segment in enumerate(transcription_result.segments, 1):
            start_time = self._seconds_to_srt_time(segment.start)
            end_time = self._seconds_to_srt_time(segment.end)
            
            lines.append(f"{i}")
            lines.append(f"{start_time} --> {end_time}")
            lines.append(segment.text)
            lines.append("")
        
        return "\n".join(lines)

    def _format_diarized_text(self, segments: List[Dict]) -> str:
        """Format diarized segments as readable text"""
        lines = []
        
        for seg in segments:
            if self.include_timestamps:
                timestamp = self._format_timestamp(seg["start"])
                lines.append(f"[{timestamp}] {seg['speaker']}: {seg['text']}")
            else:
                lines.append(f"{seg['speaker']}: {seg['text']}")
        
        return "\n".join(lines)

    def _format_segments_srt(self, segments: List[Dict]) -> str:
        """Format segments as SRT with speaker labels"""
        lines = []
        
        for i, seg in enumerate(segments, 1):
            start_time = self._seconds_to_srt_time(seg["start"])
            end_time = self._seconds_to_srt_time(seg["end"])
            
            lines.append(f"{i}")
            lines.append(f"{start_time} --> {end_time}")
            lines.append(f"{seg['speaker']}: {seg['text']}")
            lines.append("")
        
        return "\n".join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system usage"""
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        sanitized = "".join(c if c in safe_chars else "_" for c in filename)
        return sanitized[:100]

    def _write_file(self, filename: str, content: str, episode_dir: Path = None) -> Path:
        """Write content to file"""
        output_dir = episode_dir or self.base_output_dir
        filepath = output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Exported: {filepath}")
        return filepath
    
    def export_metadata(self, filename_prefix: str, episode, processing_metadata: Dict = None, episode_dir: Path = None) -> Path:
        """Export episode metadata to JSON file"""
        metadata = {
            "podcast": {
                "name": episode.podcast_name,
                "description": episode.podcast_description,
                "author": episode.podcast_author,
                "language": episode.podcast_language,
                "categories": episode.podcast_categories or []
            },
            "episode": {
                "title": episode.title,
                "number": episode.episode_number,
                "season": episode.season,
                "identifier": episode.episode_identifier,
                "published": episode.published,
                "duration": episode.duration,
                "hosts": episode.hosts or [],
                "author": episode.author,
                "summary": episode.summary,
                "subtitle": episode.subtitle,
                "description": episode.description,
                "explicit": episode.explicit,
                "episode_type": episode.episode_type,
                "categories": episode.categories or [],
                "language": episode.language,
                "audio_url": episode.audio_url,
                "episode_id": episode.episode_id
            },
            "processing": processing_metadata or {},
            "export": {
                "export_time": datetime.now().isoformat(),
                "format_version": "1.0"
            }
        }
        
        # Clean up None values
        metadata = self._clean_metadata(metadata)
        
        metadata_content = json.dumps(metadata, indent=2, ensure_ascii=False)
        metadata_file = self._write_file(f"{filename_prefix}_metadata.json", metadata_content, episode_dir)
        
        return metadata_file
    
    def _clean_metadata(self, obj):
        """Recursively remove None values from metadata"""
        if isinstance(obj, dict):
            return {k: self._clean_metadata(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [self._clean_metadata(item) for item in obj if item is not None]
        else:
            return obj