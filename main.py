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
    parser.add_argument("--cleanup", action="store_true", help="Run file cleanup to remove old audio files")
    parser.add_argument("--cleanup-dry-run", action="store_true", help="Preview what files would be deleted without actually deleting them")
    parser.add_argument("--file-stats", action="store_true", help="Show statistics about downloaded files")
    
    args = parser.parse_args()
    
    try:
        config = ConfigManager(args.config)
        pipeline = TranscriptionPipeline(config)
        
        if args.file_stats:
            # Show file statistics
            stats = pipeline.get_file_statistics()
            print(f"File Statistics:")
            print(f"  Total files: {stats.get('total_files', 0)}")
            print(f"  Total size: {stats.get('total_size_mb', 0)} MB")
            if stats.get('oldest_file'):
                print(f"  Oldest file: {stats['oldest_file']['path']} (modified: {stats['oldest_file']['modified']})")
            if stats.get('newest_file'):
                print(f"  Newest file: {stats['newest_file']['path']} (modified: {stats['newest_file']['modified']})")
                
        elif args.cleanup or args.cleanup_dry_run:
            # Run cleanup
            results = pipeline.cleanup_old_files(dry_run=args.cleanup_dry_run)
            action = "would be deleted" if args.cleanup_dry_run else "deleted"
            print(f"Cleanup complete: {len(results['deleted'])} files {action}, {len(results['retained'])} files retained")
            
        elif args.episode:
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