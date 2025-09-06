"""File pruning utility for managing podcast audio file retention"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

from ..config.config_manager import FileRetentionConfig

logger = logging.getLogger(__name__)


class PodcastFilePruner:
    """Manages retention and cleanup of downloaded podcast audio files"""
    
    def __init__(self, retention_config: FileRetentionConfig):
        self.config = retention_config
        self.downloads_dir = Path(retention_config.downloads_dir)
        
    def prune_old_files(self, dry_run: bool = False) -> Dict[str, List[str]]:
        """
        Remove podcast audio files older than the configured retention period.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting
            
        Returns:
            Dict with 'deleted' and 'retained' lists of file paths
        """
        if not self.config.enabled:
            logger.info("File retention pruning is disabled")
            return {'deleted': [], 'retained': []}
            
        if not self.downloads_dir.exists():
            logger.warning(f"Downloads directory does not exist: {self.downloads_dir}")
            return {'deleted': [], 'retained': []}
            
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        logger.info(f"Pruning audio files older than {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} "
                   f"({self.config.retention_days} days)")
        
        results = {'deleted': [], 'retained': []}
        
        # Find all audio files in downloads directory
        audio_files = self._find_audio_files()
        
        if not audio_files:
            logger.info("No audio files found in downloads directory")
            return results
        
        logger.info(f"Found {len(audio_files)} audio files to evaluate")
        
        for file_path in audio_files:
            try:
                # Get file modification time (when it was downloaded)
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    # File is older than retention period
                    if dry_run:
                        logger.info(f"[DRY RUN] Would delete: {file_path} (modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                    else:
                        try:
                            file_path.unlink()
                            logger.info(f"Deleted: {file_path} (modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                        except OSError as e:
                            logger.error(f"Failed to delete {file_path}: {e}")
                            continue
                    
                    results['deleted'].append(str(file_path))
                else:
                    # File is within retention period
                    logger.debug(f"Retained: {file_path} (modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')})")
                    results['retained'].append(str(file_path))
                    
            except (OSError, ValueError) as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Log summary
        deleted_count = len(results['deleted'])
        retained_count = len(results['retained'])
        
        if dry_run:
            logger.info(f"Pruning summary (DRY RUN): {deleted_count} files would be deleted, {retained_count} files retained")
        else:
            logger.info(f"Pruning summary: {deleted_count} files deleted, {retained_count} files retained")
            
        return results
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the downloads directory"""
        audio_files = []
        
        for ext in self.config.audio_extensions:
            # Use glob to find files with each extension (case-insensitive)
            pattern = f"*{ext}"
            audio_files.extend(self.downloads_dir.glob(pattern))
            # Also check uppercase extensions
            pattern_upper = f"*{ext.upper()}"
            audio_files.extend(self.downloads_dir.glob(pattern_upper))
        
        # Remove duplicates and sort by modification time (oldest first)
        unique_files = list(set(audio_files))
        unique_files.sort(key=lambda f: f.stat().st_mtime)
        
        return unique_files
    
    def get_file_stats(self) -> Dict:
        """Get statistics about files in the downloads directory"""
        if not self.downloads_dir.exists():
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_file': None,
                'newest_file': None
            }
        
        audio_files = self._find_audio_files()
        
        if not audio_files:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_file': None,
                'newest_file': None
            }
        
        total_size = sum(f.stat().st_size for f in audio_files)
        oldest_file = min(audio_files, key=lambda f: f.stat().st_mtime)
        newest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
        
        return {
            'total_files': len(audio_files),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'oldest_file': {
                'path': str(oldest_file),
                'modified': datetime.fromtimestamp(oldest_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            },
            'newest_file': {
                'path': str(newest_file),
                'modified': datetime.fromtimestamp(newest_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def cleanup_empty_directories(self, dry_run: bool = False) -> List[str]:
        """
        Remove empty subdirectories in the downloads folder after pruning.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of directories that were (or would be) removed
        """
        if not self.downloads_dir.exists():
            return []
        
        removed_dirs = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.downloads_dir, topdown=False):
            root_path = Path(root)
            
            # Skip the main downloads directory
            if root_path == self.downloads_dir:
                continue
                
            try:
                # Check if directory is empty
                if not any(root_path.iterdir()):
                    if dry_run:
                        logger.info(f"[DRY RUN] Would remove empty directory: {root_path}")
                    else:
                        root_path.rmdir()
                        logger.info(f"Removed empty directory: {root_path}")
                    
                    removed_dirs.append(str(root_path))
                    
            except OSError as e:
                logger.error(f"Error checking/removing directory {root_path}: {e}")
                
        return removed_dirs