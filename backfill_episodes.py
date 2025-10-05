#!/usr/bin/env python3
"""
Backfill script to process older Intelligent Machines episodes (805-825)
"""

import requests
import sys
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config.config_manager import ConfigManager
from src.transcription.pipeline import TranscriptionPipeline
from src.utils.rss_parser import Episode

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scrape_episode_info(episode_number: int) -> Episode:
    """Scrape episode information from TWiT website"""
    url = f"https://twit.tv/shows/intelligent-machines/episodes/{episode_number}"
    
    logger.info(f"Scraping episode info for IM {episode_number}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else f"IM {episode_number}"
        
        # Clean up title - remove episode number prefix if present
        if title.startswith(f"IM {episode_number}"):
            title = title[len(f"IM {episode_number}"):].strip(" :-")
        # Don't modify the title - keep it clean for directory naming
        
        # Extract description
        description_elem = soup.find('meta', {'name': 'description'}) or soup.find('p')
        description = description_elem.get('content', '') if description_elem and description_elem.get('content') else \
                     description_elem.get_text().strip() if description_elem else ""
        
        # Look for audio file link
        audio_url = None
        
        # Try to find MP3 links
        audio_links = soup.find_all('a', href=True)
        for link in audio_links:
            href = link['href']
            if href.endswith('.mp3') or 'mp3' in href.lower():
                audio_url = href
                if not audio_url.startswith('http'):
                    audio_url = urljoin(url, audio_url)
                break
        
        # If no direct MP3 link found, try alternative approaches
        if not audio_url:
            # Look for audio elements
            audio_elem = soup.find('audio')
            if audio_elem and audio_elem.get('src'):
                audio_url = urljoin(url, audio_elem['src'])
            
            # Look for download links
            if not audio_url:
                download_links = soup.find_all('a', string=lambda text: text and 'download' in text.lower())
                for link in download_links:
                    if link.get('href') and ('.mp3' in link['href'] or 'audio' in link['href']):
                        audio_url = urljoin(url, link['href'])
                        break
        
        # If still no audio URL, construct a likely URL based on pattern
        if not audio_url:
            # Try common TWiT audio URL patterns
            potential_urls = [
                f"https://cdn.twit.tv/audio/twig/twig{episode_number:04d}/twig{episode_number:04d}.mp3",
                f"https://cdn.twit.tv/audio/im/im{episode_number:04d}/im{episode_number:04d}.mp3",
                f"https://cdn.twit.tv/libsyn/twig_{episode_number}/R1_twig{episode_number:04d}.mp3"
            ]
            
            for test_url in potential_urls:
                try:
                    test_response = requests.head(test_url, timeout=10)
                    if test_response.status_code == 200:
                        audio_url = test_url
                        logger.info(f"Found audio at: {audio_url}")
                        break
                except:
                    continue
        
        if not audio_url:
            raise ValueError(f"Could not find audio URL for episode {episode_number}")
        
        return Episode(
            title=title,
            description=description,
            audio_url=audio_url,
            published="",
            episode_id=str(episode_number),
            podcast_name="Intelligent_Machines",
            episode_identifier=f"IM_{episode_number}",
            episode_number=str(episode_number),
            podcast_description="Meet the AI pioneers, inventors, and innovators who are about to disrupt every aspect of modern life.",
            podcast_author="TWiT",
            hosts=["Leo Laporte", "Jeff Jarvis", "Paris Martineau"]
        )
        
    except Exception as e:
        logger.error(f"Error scraping episode {episode_number}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Backfill older Intelligent Machines episodes")
    parser.add_argument("--start", "-s", type=int, default=805, help="Starting episode number")
    parser.add_argument("--end", "-e", type=int, default=825, help="Ending episode number")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--test", "-t", action="store_true", help="Test mode - only process first 2 episodes")
    parser.add_argument("--dry-run", action="store_true", help="Dry run - don't actually process episodes")
    
    args = parser.parse_args()
    
    if args.test:
        args.end = min(args.start + 2, args.end)
        logger.info(f"Test mode: only processing episodes {args.start}-{args.end}")
    
    try:
        # Initialize the pipeline
        config = ConfigManager(args.config)
        pipeline = TranscriptionPipeline(config)
        
        logger.info(f"Starting backfill for episodes {args.start} to {args.end}")
        
        success_count = 0
        error_count = 0
        
        for episode_num in range(args.start, args.end + 1):
            try:
                logger.info(f"Processing episode IM {episode_num}")
                
                # Check if episode already exists
                existing_dirs = list(Path("output/Intelligent_Machines_Audio").glob(f"IM_{episode_num}_*"))
                if existing_dirs:
                    logger.info(f"Episode IM {episode_num} already processed, skipping")
                    continue
                
                # Scrape episode information
                episode = scrape_episode_info(episode_num)
                
                if args.dry_run:
                    logger.info(f"DRY RUN: Would process {episode.title}")
                    logger.info(f"  Audio URL: {episode.audio_url}")
                    success_count += 1
                    continue
                
                # Process the episode
                result = pipeline.process_episode(episode)
                
                if result:
                    logger.info(f"Successfully processed IM {episode_num}")
                    success_count += 1
                else:
                    logger.error(f"Failed to process IM {episode_num}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing episode {episode_num}: {e}")
                error_count += 1
                continue
        
        logger.info(f"Backfill complete: {success_count} successful, {error_count} errors")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()