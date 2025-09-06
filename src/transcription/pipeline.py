"""Main transcription pipeline orchestrating all components"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config import ConfigManager
from ..utils import RSSParser, Episode
from ..transcription import get_transcriber
from ..transcription.whisperx_transcriber import WhisperXTranscriber
from ..diarization import get_diarizer
from ..llm import SpeakerIdentifier
from ..export import TranscriptExporter

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.get_config()
        self.rss_parser = RSSParser()
        
        # Use WhisperX for better diarization
        self.whisperx_transcriber = WhisperXTranscriber(
            model_size=self.config.transcription.model_size,
            device=self.config.transcription.device,
            hf_token=self.config.diarization.hf_token
        )
        
        # Only initialize speaker identifier if API key is available
        if self.config.llm.api_key:
            self.speaker_identifier = SpeakerIdentifier(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
        else:
            from ..llm import MockSpeakerIdentifier
            self.speaker_identifier = MockSpeakerIdentifier()
            logger.info("No API key found - using mock speaker identifier")
        
        self.exporter = TranscriptExporter(
            output_dir=self.config.output.base_dir,
            include_timestamps=self.config.output.include_timestamps
        )

    def process_all_feeds(self, output_dir: Optional[str] = None):
        """Process all configured RSS feeds"""
        logger.info("Starting batch processing of all feeds")
        
        for feed_config in self.config.feeds:
            if not feed_config.enabled:
                logger.info(f"Skipping disabled feed: {feed_config.name}")
                continue
            
            try:
                self.process_feed(feed_config.url, output_dir, feed_config.max_episodes)
            except Exception as e:
                logger.error(f"Error processing feed {feed_config.name}: {e}")

    def process_feed(self, feed_url: str, output_dir: Optional[str] = None, 
                    max_episodes: Optional[int] = None):
        """Process all episodes from a specific RSS feed"""
        logger.info(f"Processing feed: {feed_url}")
        
        episodes = self.rss_parser.parse_feed(feed_url, max_episodes)
        
        if not episodes:
            logger.warning(f"No episodes found in feed: {feed_url}")
            return
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for episode in episodes:
                future = executor.submit(self.process_episode, episode, output_dir)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Episode processing failed: {e}")

    def process_single_episode(self, episode_url: str, output_dir: Optional[str] = None):
        """Process a single episode from URL"""
        episode = Episode(
            title="Single Episode",
            description="",
            audio_url=episode_url,
            published=""
        )
        
        self.process_episode(episode, output_dir)

    def process_episode(self, episode: Episode, output_dir: Optional[str] = None):
        """Process a single episode through the complete pipeline"""
        start_time = time.time()
        logger.info(f"Processing episode: {episode.title}")
        
        # Check if episode already has transcription files
        if self._episode_already_transcribed(episode, output_dir):
            logger.info(f"Episode already transcribed, skipping: {episode.title}")
            return None
        
        try:
            audio_path = self.rss_parser.download_audio(episode, 
                                                       Path(output_dir) if output_dir else None)
            if not audio_path:
                raise RuntimeError("Failed to download audio")
            
            logger.info("Step 1-2: Running WhisperX transcription and diarization...")
            logger.info(f"About to call WhisperX with audio path: {audio_path}")
            try:
                whisperx_result = self.whisperx_transcriber.transcribe_and_diarize(audio_path)
                logger.info("WhisperX completed successfully")
            except Exception as e:
                logger.error(f"WhisperX failed with error: {e}")
                logger.error(f"Error type: {type(e)}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise
            
            raw_transcription = whisperx_result["raw_transcription"]
            diarized_segments = whisperx_result["diarized_segments"]
            
            logger.info("Step 3: Identifying speakers with LLM...")
            llm_segments, speaker_mappings = self.speaker_identifier.identify_speakers(diarized_segments)
            
            logger.info("Step 4: Exporting transcripts and metadata...")
            
            # Prepare processing metadata
            processing_metadata = {
                "transcribed_date": datetime.now().isoformat(),
                "model_used": self.config.transcription.model_size,
                "language_detected": whisperx_result.get("language", "unknown"),
                "device_used": self.config.transcription.device,
                "diarization_enabled": self.config.diarization.enabled,
                "llm_provider": self.config.llm.provider if hasattr(self.config.llm, 'provider') else None,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
            
            output_files = self.exporter.export_all_formats(
                episode,
                raw_transcription,
                diarized_segments,
                llm_segments,
                speaker_mappings,
                processing_metadata
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Episode processed successfully in {processing_time:.1f}s: {episode.title}")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Failed to process episode {episode.title}: {e}")
            raise

    def _episode_already_transcribed(self, episode: Episode, output_dir: Optional[str] = None) -> bool:
        """Check if episode already has transcription files"""
        # Use the same path generation logic as the exporter
        base_output_dir = Path(output_dir) if output_dir else Path(self.config.output.base_dir)
        
        safe_title = self.exporter._sanitize_filename(episode.title)
        safe_podcast_name = self.exporter._sanitize_filename(episode.podcast_name or "Unknown_Podcast")
        
        episode_dir = base_output_dir / safe_podcast_name / safe_title
        
        # Use episode identifier for filename, fallback to safe title
        filename_prefix = episode.episode_identifier or safe_title
        
        # Check for any of the main transcription files
        transcription_files = [
            f"{filename_prefix}_raw.json",
            f"{filename_prefix}_diarized.json", 
            f"{filename_prefix}_enhanced.json"
        ]
        
        for filename in transcription_files:
            filepath = episode_dir / filename
            if filepath.exists():
                logger.debug(f"Found existing transcription file: {filepath}")
                return True
        
        return False

    def get_processing_stats(self) -> Dict:
        """Get statistics about processing capabilities"""
        return {
            "transcription_model": self.config.transcription.model_size,
            "diarization_enabled": self.config.diarization.enabled,
            "llm_provider": self.config.llm.provider,
            "output_formats": self.config.output.formats,
            "feeds_configured": len(self.config.feeds)
        }