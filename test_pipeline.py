#!/usr/bin/env python3
"""Test pipeline for AI Auto Transcripts"""

import logging
import sys
from pathlib import Path

from src.config.config_manager import ConfigManager
from src.transcription.pipeline import TranscriptionPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_single_episode():
    """Test processing a single episode"""
    logger.info("Testing single episode processing")
    
    test_url = "https://example.com/test-episode.mp3"
    
    try:
        config = ConfigManager("config.yaml")
        pipeline = TranscriptionPipeline(config)
        
        logger.info("Pipeline initialized successfully")
        stats = pipeline.get_processing_stats()
        logger.info(f"Processing stats: {stats}")
        
        logger.info("Note: To test with real audio, replace test_url with actual episode URL")
        logger.info(f"Would process: {test_url}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)


def test_feed_parsing():
    """Test RSS feed parsing without processing audio"""
    logger.info("Testing RSS feed parsing")
    
    try:
        config = ConfigManager("config.yaml")
        pipeline = TranscriptionPipeline(config)
        
        test_feed = "https://feeds.feedburner.com/JoeRogansTheJoeRoganExperience"
        episodes = pipeline.rss_parser.parse_feed(test_feed, max_episodes=2)
        
        logger.info(f"Found {len(episodes)} episodes in test feed")
        for i, episode in enumerate(episodes, 1):
            logger.info(f"Episode {i}: {episode.title[:50]}...")
            logger.info(f"  Audio URL: {episode.audio_url}")
        
    except Exception as e:
        logger.error(f"Feed parsing test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("Starting AI Auto Transcripts test pipeline")
    
    test_feed_parsing()
    test_single_episode()
    
    logger.info("Test pipeline completed successfully")