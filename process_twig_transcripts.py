#!/usr/bin/env python3
"""
Process TWIG (This Week in Google) transcripts from external GitHub repository.
Fetches diarized transcripts from wesgould/twit-transcripts and applies LLM enhancement.

Usage:
    # Process all available TWIG episodes
    python process_twig_transcripts.py

    # Process specific episode range
    python process_twig_transcripts.py --start 700 --end 725

    # Process single episode
    python process_twig_transcripts.py --episode 725

    # List available episodes (dry run)
    python process_twig_transcripts.py --list
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv

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
        logging.FileHandler('process_twig_transcripts.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# GitHub repository configuration
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/wesgould/twit-transcripts/main/twig/diarized"
GITHUB_API_BASE = "https://api.github.com/repos/wesgould/twit-transcripts/contents/twig/diarized"


class TWIGTranscriptParser:
    """Parse TWIG diarized transcript format into segments."""

    @staticmethod
    def parse_transcript(content: str) -> List[Dict]:
        """
        Parse TWIG diarized transcript format.

        Format:
            Start time: 0.34
            End time: 159.50
            Speaker: Leo Laporte
            Transcript: It's time for Twig...

        Returns list of segment dicts with start, end, speaker, text.
        """
        segments = []

        # Split into blocks (separated by blank lines)
        blocks = re.split(r'\n\s*\n', content)

        segment_index = 0
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Skip metadata header (FFMETADATA1 section)
            if block.startswith(';FFMETADATA') or block.startswith('['):
                continue

            # Parse the segment
            segment = TWIGTranscriptParser._parse_segment_block(block, segment_index)
            if segment:
                segments.append(segment)
                segment_index += 1

        return segments

    @staticmethod
    def _parse_segment_block(block: str, segment_index: int = 0) -> Optional[Dict]:
        """Parse a single segment block. Handles both timestamped and non-timestamped formats."""
        speaker_match = re.search(r'Speaker:\s*(.+?)(?:\n|$)', block)
        transcript_match = re.search(r'Transcript:\s*(.+)', block, re.DOTALL)

        if not speaker_match or not transcript_match:
            return None

        # Try to get timestamps (newer format)
        start_match = re.search(r'Start time:\s*([\d.]+)', block)
        end_match = re.search(r'End time:\s*([\d.]+)', block)

        if start_match and end_match:
            start = float(start_match.group(1))
            end = float(end_match.group(1))
        else:
            # Older format without timestamps - use segment index as placeholder
            start = float(segment_index * 30)  # Approximate 30 sec per segment
            end = float((segment_index + 1) * 30)

        return {
            "start": start,
            "end": end,
            "speaker": speaker_match.group(1).strip(),
            "text": transcript_match.group(1).strip()
        }


class TWIGTranscriptProcessor:
    """Process TWIG transcripts from GitHub repository."""

    def __init__(self, config_path: str = "config.yaml", output_dir: str = None):
        self.config = ConfigManager(config_path).get_config()

        # Allow override of output directory
        base_dir = output_dir or self.config.output.base_dir

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
            logger.warning("No API key found - using mock speaker identifier")

        # Initialize exporter
        self.exporter = TranscriptExporter(
            output_dir=base_dir,
            include_timestamps=self.config.output.include_timestamps
        )
        self.base_output_dir = Path(base_dir)

    def list_available_episodes(self) -> List[int]:
        """List all available TWIG episode numbers from GitHub."""
        logger.info("Fetching available episodes from GitHub...")

        try:
            response = requests.get(GITHUB_API_BASE)
            response.raise_for_status()
            files = response.json()

            episodes = []
            for file_info in files:
                name = file_info.get('name', '')
                match = re.match(r'twig(\d+)-d\.txt', name)
                if match:
                    episodes.append(int(match.group(1)))

            episodes.sort()
            return episodes

        except requests.RequestException as e:
            logger.error(f"Failed to fetch episode list: {e}")
            return []

    def fetch_transcript(self, episode_num: int) -> Optional[str]:
        """Fetch transcript content from GitHub."""
        filename = f"twig{episode_num:04d}-d.txt"
        url = f"{GITHUB_RAW_BASE}/{filename}"

        logger.info(f"Fetching {url}...")

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch episode {episode_num}: {e}")
            return None

    def create_episode_metadata(self, episode_num: int) -> Dict:
        """Create metadata for TWIG episode to help with speaker identification."""
        return {
            "podcast": {
                "name": "This Week in Google",
                "hosts": ["Leo Laporte", "Jeff Jarvis", "Paris Martineau", "Ant Pruitt"]
            },
            "episode": {
                "title": f"TWIG {episode_num}",
                "number": episode_num,
                "description": "This Week in Google - a weekly roundtable discussion about Google, technology, and the internet."
            }
        }

    def process_episode(self, episode_num: int, skip_existing: bool = True) -> Dict:
        """Process a single TWIG episode."""
        start_time = time.time()
        episode_id = f"TWIG_{episode_num:04d}"
        logger.info(f"Processing {episode_id}...")

        # Check if already processed
        episode_dir = self.base_output_dir / "This_Week_in_Google" / episode_id
        enhanced_file = episode_dir / f"{episode_id}_enhanced.json"

        if skip_existing and enhanced_file.exists():
            logger.info(f"Skipping {episode_id} - already processed")
            return {
                'success': True,
                'episode': episode_id,
                'skipped': True,
                'reason': 'already_processed'
            }

        try:
            # Fetch transcript
            content = self.fetch_transcript(episode_num)
            if not content:
                return {
                    'success': False,
                    'episode': episode_id,
                    'error': 'Failed to fetch transcript'
                }

            # Parse transcript
            segments = TWIGTranscriptParser.parse_transcript(content)
            if not segments:
                return {
                    'success': False,
                    'episode': episode_id,
                    'error': 'Failed to parse transcript - no segments found'
                }

            logger.info(f"Parsed {len(segments)} segments")

            # Show unique speakers found
            unique_speakers = set(seg['speaker'] for seg in segments)
            logger.info(f"Unique speakers: {sorted(unique_speakers)}")

            # Create metadata for speaker identification
            metadata = self.create_episode_metadata(episode_num)

            # Apply LLM speaker identification
            logger.info("Applying LLM speaker identification...")
            llm_segments, speaker_mappings = self.speaker_identifier.identify_speakers(
                segments, metadata
            )

            if speaker_mappings:
                logger.info(f"Speaker mappings: {speaker_mappings}")

            # Create episode-specific output directory
            episode_dir.mkdir(parents=True, exist_ok=True)

            # Export diarized transcript (original)
            self._export_diarized(episode_id, segments, episode_dir)

            # Export LLM-enhanced transcript
            self.exporter.export_llm_transcript(
                episode_id, llm_segments, speaker_mappings, episode_dir
            )

            # Export metadata
            self._export_metadata(episode_id, episode_num, speaker_mappings,
                                 time.time() - start_time, episode_dir)

            processing_time = time.time() - start_time
            logger.info(f"Processed {episode_id} in {processing_time:.1f}s")

            return {
                'success': True,
                'episode': episode_id,
                'segments': len(segments),
                'speaker_mappings': speaker_mappings,
                'processing_time': processing_time
            }

        except Exception as e:
            logger.error(f"Failed to process {episode_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'episode': episode_id,
                'error': str(e)
            }

    def _export_diarized(self, episode_id: str, segments: List[Dict], episode_dir: Path):
        """Export diarized transcript (before LLM enhancement)."""
        # Text format
        txt_content = []
        for seg in segments:
            timestamp = self._format_timestamp(seg["start"])
            txt_content.append(f"[{timestamp}] {seg['speaker']}: {seg['text']}")

        txt_file = episode_dir / f"{episode_id}_diarized.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_content))

        # JSON format
        json_content = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "type": "diarized_transcript",
                "source": "twit-transcripts GitHub repository",
                "num_segments": len(segments)
            },
            "segments": segments
        }
        json_file = episode_dir / f"{episode_id}_diarized.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, indent=2)

        logger.info(f"Exported diarized: {txt_file}")

    def _export_metadata(self, episode_id: str, episode_num: int,
                        speaker_mappings: Dict, processing_time: float, episode_dir: Path):
        """Export processing metadata."""
        metadata = {
            "podcast": {
                "name": "This Week in Google",
                "abbreviation": "TWIG"
            },
            "episode": {
                "title": f"TWIG {episode_num}",
                "number": episode_num,
                "identifier": episode_id
            },
            "processing": {
                "processed_date": datetime.now().isoformat(),
                "source": "wesgould/twit-transcripts GitHub repository",
                "source_url": f"{GITHUB_RAW_BASE}/twig{episode_num:04d}-d.txt",
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "processing_time_seconds": round(processing_time, 2),
                "speaker_mappings": speaker_mappings or {}
            }
        }

        metadata_file = episode_dir / f"{episode_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported metadata: {metadata_file}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def process_episodes(self, episodes: List[int], skip_existing: bool = True) -> List[Dict]:
        """Process multiple episodes."""
        results = []
        total_start = time.time()

        for i, episode_num in enumerate(episodes, 1):
            logger.info(f"Processing episode {i}/{len(episodes)}: TWIG {episode_num}")
            result = self.process_episode(episode_num, skip_existing=skip_existing)
            results.append(result)

            # Rate limiting pause between API calls
            if result.get('success') and not result.get('skipped'):
                time.sleep(1)

        # Print summary
        total_time = time.time() - total_start
        successful = [r for r in results if r['success'] and not r.get('skipped')]
        skipped = [r for r in results if r.get('skipped')]
        failed = [r for r in results if not r['success']]

        logger.info(f"\n{'='*50}")
        logger.info("PROCESSING COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Total episodes: {len(episodes)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Skipped: {len(skipped)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total time: {total_time:.1f}s")

        if failed:
            logger.info(f"\nFailed episodes:")
            for r in failed:
                logger.info(f"  - {r['episode']}: {r.get('error', 'Unknown error')}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Process TWIG transcripts from GitHub with LLM speaker identification"
    )
    parser.add_argument(
        '--episode', '-e',
        type=int,
        help='Process a single episode number'
    )
    parser.add_argument(
        '--start', '-s',
        type=int,
        help='Starting episode number (inclusive)'
    )
    parser.add_argument(
        '--end', '-n',
        type=int,
        help='Ending episode number (inclusive)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available episodes without processing'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force reprocessing of already processed episodes'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory (default: from config.yaml)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    processor = TWIGTranscriptProcessor(
        config_path=args.config,
        output_dir=args.output
    )

    if args.list:
        # List available episodes
        episodes = processor.list_available_episodes()
        if episodes:
            print(f"\nAvailable TWIG episodes: {len(episodes)}")
            print(f"Range: {min(episodes)} - {max(episodes)}")
            print(f"\nEpisode numbers: {episodes[:10]}...{episodes[-10:]}")
        else:
            print("No episodes found or failed to fetch list")
        return

    # Determine which episodes to process
    if args.episode:
        episodes = [args.episode]
    elif args.start is not None or args.end is not None:
        available = processor.list_available_episodes()
        start = args.start or min(available)
        end = args.end or max(available)
        episodes = [ep for ep in available if start <= ep <= end]
    else:
        # Default: process all available
        episodes = processor.list_available_episodes()

    if not episodes:
        logger.error("No episodes to process")
        sys.exit(1)

    logger.info(f"Will process {len(episodes)} episodes")

    # Process episodes
    results = processor.process_episodes(
        episodes,
        skip_existing=not args.force
    )

    # Exit with error code if any failures
    failed = [r for r in results if not r['success']]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
