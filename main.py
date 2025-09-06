#!/usr/bin/env python3
"""
AI Auto Transcripts - Podcast transcription with speaker diarization and identification
"""

import argparse
import sys
from pathlib import Path

from src.config.config_manager import ConfigManager
from src.transcription.pipeline import TranscriptionPipeline


def main():
    parser = argparse.ArgumentParser(description="AI Auto Transcripts - Podcast transcription system")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--episode", "-e", type=str, help="Process single episode URL")
    parser.add_argument("--feed", "-f", type=str, help="Process specific RSS feed")
    parser.add_argument("--output-dir", "-o", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    try:
        config = ConfigManager(args.config)
        pipeline = TranscriptionPipeline(config)
        
        if args.episode:
            pipeline.process_single_episode(args.episode, args.output_dir)
        elif args.feed:
            pipeline.process_feed(args.feed, args.output_dir)
        else:
            pipeline.process_all_feeds(args.output_dir)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()