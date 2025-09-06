#!/usr/bin/env python3
"""
Reprocess existing transcripts with new LLM speaker identification capability
Processes all existing episodes through the updated LLM enhancement system
"""

import json
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config_manager import ConfigManager
from src.llm.speaker_identifier import SpeakerIdentifier, MockSpeakerIdentifier
from src.export.transcript_exporter import TranscriptExporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reprocess_transcripts.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TranscriptReprocessor:
    def __init__(self, config_path="config.yaml"):
        self.config = ConfigManager(config_path).get_config()
        
        # Initialize LLM speaker identifier
        if self.config.llm.api_key:
            self.speaker_identifier = SpeakerIdentifier(
                provider=self.config.llm.provider,
                model=self.config.llm.model,
                api_key=self.config.llm.api_key,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens
            )
            logger.info(f"Using {self.config.llm.provider} {self.config.llm.model} for speaker identification")
        else:
            self.speaker_identifier = MockSpeakerIdentifier()
            logger.info("No API key found - using mock speaker identifier")
        
        # Initialize exporter
        self.exporter = TranscriptExporter(
            output_dir=self.config.output.base_dir,
            include_timestamps=self.config.output.include_timestamps
        )
    
    def find_existing_episodes(self):
        """Find all existing episode directories with diarized transcripts"""
        output_dir = Path(self.config.output.base_dir)
        episodes = []
        
        for podcast_dir in output_dir.iterdir():
            if podcast_dir.is_dir():
                for episode_dir in podcast_dir.iterdir():
                    if episode_dir.is_dir():
                        # Look for diarized JSON file
                        diarized_files = list(episode_dir.glob("*_diarized.json"))
                        if diarized_files:
                            episodes.append({
                                'podcast_name': podcast_dir.name,
                                'episode_name': episode_dir.name,
                                'episode_dir': episode_dir,
                                'diarized_file': diarized_files[0]
                            })
        
        return episodes
    
    def load_diarized_transcript(self, json_path):
        """Load diarized transcript from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['segments']
    
    def create_episode_object(self, episode_info):
        """Create a minimal episode object for the exporter"""
        class Episode:
            def __init__(self, title, podcast_name, episode_identifier):
                self.title = title
                self.podcast_name = podcast_name
                self.episode_identifier = episode_identifier
                self.description = ""
                self.audio_url = ""
                self.published = ""
        
        # Extract episode identifier from filename
        diarized_filename = episode_info['diarized_file'].stem
        episode_id = diarized_filename.replace('_diarized', '')
        
        return Episode(
            title=episode_info['episode_name'].replace('_', ' '),
            podcast_name=episode_info['podcast_name'].replace('_Audio', '').replace('_', ' '),
            episode_identifier=episode_id
        )
    
    def reprocess_episode(self, episode_info):
        """Reprocess a single episode with new LLM enhancement"""
        start_time = time.time()
        logger.info(f"Reprocessing: {episode_info['podcast_name']} - {episode_info['episode_name']}")
        
        try:
            # Load existing diarized transcript
            diarized_segments = self.load_diarized_transcript(episode_info['diarized_file'])
            logger.info(f"Loaded {len(diarized_segments)} diarized segments")
            
            # Show unique speakers found
            unique_speakers = set(seg['speaker'] for seg in diarized_segments)
            logger.info(f"Unique speakers: {sorted(unique_speakers)}")
            
            # Apply LLM speaker identification
            logger.info("Applying LLM speaker identification...")
            
            # Try to load existing metadata for better speaker identification
            metadata = None
            metadata_file = episode_info['diarized_file'].parent / f"{episode_info['diarized_file'].stem.replace('_diarized', '_metadata')}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    logger.info("Loaded existing metadata for speaker identification")
                except Exception as e:
                    logger.warning(f"Failed to load metadata: {e}")
            
            llm_segments, speaker_mappings = self.speaker_identifier.identify_speakers(diarized_segments, metadata)
            
            if speaker_mappings:
                logger.info(f"Speaker mappings found: {speaker_mappings}")
            else:
                logger.info("No speaker mappings found or LLM identification failed")
            
            # Create episode object for exporter
            episode = self.create_episode_object(episode_info)
            
            # Create processing metadata
            processing_metadata = {
                "reprocessed_date": datetime.now().isoformat(),
                "original_transcription_date": "unknown",
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "reprocessing_time_seconds": round(time.time() - start_time, 2),
                "speaker_mappings_applied": speaker_mappings or {}
            }
            
            # Export only the enhanced transcript (don't overwrite raw/diarized)
            episode_dir = episode_info['episode_dir']
            filename_prefix = episode.episode_identifier
            
            enhanced_file = self.exporter.export_llm_transcript(
                filename_prefix, 
                llm_segments, 
                speaker_mappings, 
                episode_dir
            )
            
            # Update metadata file
            metadata_file = episode_dir / f"{filename_prefix}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                existing_metadata.update(processing_metadata)
            else:
                existing_metadata = processing_metadata
            
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Reprocessed successfully in {processing_time:.1f}s")
            
            return {
                'success': True,
                'episode': episode_info['episode_name'],
                'speaker_mappings': speaker_mappings,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to reprocess {episode_info['episode_name']}: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            return {
                'success': False,
                'episode': episode_info['episode_name'],
                'error': str(e)
            }
    
    def reprocess_all(self):
        """Reprocess all existing episodes"""
        episodes = self.find_existing_episodes()
        
        if not episodes:
            logger.info("No existing episodes found to reprocess")
            return
        
        logger.info(f"Found {len(episodes)} episodes to reprocess:")
        for ep in episodes:
            logger.info(f"  - {ep['podcast_name']}: {ep['episode_name']}")
        
        results = []
        total_start = time.time()
        
        for episode_info in episodes:
            result = self.reprocess_episode(episode_info)
            results.append(result)
            
            # Brief pause between episodes to be nice to the API
            time.sleep(1)
        
        # Summary
        total_time = time.time() - total_start
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"REPROCESSING COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Total episodes: {len(episodes)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total time: {total_time:.1f}s")
        
        if successful:
            logger.info(f"\n✅ Successfully reprocessed:")
            for result in successful:
                mappings = result.get('speaker_mappings', {})
                mapping_info = f" ({len(mappings)} speakers mapped)" if mappings else " (no mappings)"
                logger.info(f"  - {result['episode']}{mapping_info}")
        
        if failed:
            logger.info(f"\n❌ Failed to reprocess:")
            for result in failed:
                logger.info(f"  - {result['episode']}: {result['error']}")


def main():
    logger.info("Starting transcript reprocessing with new LLM capability")
    
    try:
        reprocessor = TranscriptReprocessor()
        reprocessor.reprocess_all()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()