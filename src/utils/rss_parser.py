"""RSS feed parsing and episode extraction"""

import feedparser
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from urllib.parse import urljoin, urlparse
import time
import re

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    title: str
    description: str
    audio_url: str
    published: str
    duration: Optional[int] = None
    episode_id: Optional[str] = None
    podcast_name: Optional[str] = None
    episode_identifier: Optional[str] = None
    # Additional metadata fields
    episode_number: Optional[str] = None
    season: Optional[str] = None
    hosts: Optional[List[str]] = None
    author: Optional[str] = None
    summary: Optional[str] = None
    subtitle: Optional[str] = None
    explicit: Optional[bool] = None
    episode_type: Optional[str] = None
    categories: Optional[List[str]] = None
    language: Optional[str] = None
    # Feed-level metadata
    podcast_description: Optional[str] = None
    podcast_author: Optional[str] = None
    podcast_language: Optional[str] = None
    podcast_categories: Optional[List[str]] = None


class RSSParser:
    def __init__(self, download_dir: str = "downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

    def parse_feed(self, feed_url: str, max_episodes: Optional[int] = None) -> List[Episode]:
        """Parse RSS feed and extract episode information"""
        try:
            logger.info(f"Parsing RSS feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"Feed parsing had issues: {feed.bozo_exception}")
            
            # Extract podcast-level metadata
            podcast_name = self._extract_podcast_name(feed)
            podcast_metadata = self._extract_podcast_metadata(feed)
            logger.info(f"Extracted podcast name: {podcast_name}")
            
            episodes = []
            entries = feed.entries[:max_episodes] if max_episodes else feed.entries
            
            for entry in entries:
                audio_url = self._extract_audio_url(entry)
                if not audio_url:
                    logger.warning(f"No audio URL found for episode: {entry.get('title', 'Unknown')}")
                    continue
                
                episode_title = entry.get('title', 'Unknown Title')
                episode_identifier = self._extract_episode_identifier(episode_title)
                episode_metadata = self._extract_episode_metadata(entry)
                
                episode = Episode(
                    title=episode_title,
                    description=entry.get('description', ''),
                    audio_url=audio_url,
                    published=entry.get('published', ''),
                    episode_id=entry.get('id', entry.get('link', '')),
                    podcast_name=podcast_name,
                    episode_identifier=episode_identifier,
                    # Episode-specific metadata
                    episode_number=episode_metadata.get('episode_number'),
                    season=episode_metadata.get('season'),
                    hosts=episode_metadata.get('hosts'),
                    author=episode_metadata.get('author'),
                    summary=episode_metadata.get('summary'),
                    subtitle=episode_metadata.get('subtitle'),
                    explicit=episode_metadata.get('explicit'),
                    episode_type=episode_metadata.get('episode_type'),
                    categories=episode_metadata.get('categories'),
                    language=episode_metadata.get('language'),
                    # Podcast-level metadata
                    podcast_description=podcast_metadata.get('description'),
                    podcast_author=podcast_metadata.get('author'),
                    podcast_language=podcast_metadata.get('language'),
                    podcast_categories=podcast_metadata.get('categories')
                )
                
                if hasattr(entry, 'itunes_duration'):
                    episode.duration = self._parse_duration(entry.itunes_duration)
                
                episodes.append(episode)
            
            logger.info(f"Found {len(episodes)} episodes with audio")
            return episodes
            
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {e}")
            return []

    def _extract_audio_url(self, entry) -> Optional[str]:
        """Extract audio URL from RSS entry"""
        if hasattr(entry, 'enclosures'):
            for enclosure in entry.enclosures:
                if enclosure.type and ('audio' in enclosure.type.lower() or 
                                     enclosure.href.lower().endswith(('.mp3', '.wav', '.m4a'))):
                    return enclosure.href
        
        if hasattr(entry, 'links'):
            for link in entry.links:
                if link.type and 'audio' in link.type.lower():
                    return link.href
                if link.href.lower().endswith(('.mp3', '.wav', '.m4a')):
                    return link.href
        
        return None

    def _parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse duration string to seconds"""
        try:
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return int(duration_str)
        except (ValueError, IndexError):
            return None

    def download_audio(self, episode: Episode, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Download audio file from episode"""
        output_dir = output_dir or self.download_dir
        output_dir.mkdir(exist_ok=True)
        
        try:
            safe_title = "".join(c for c in episode.title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title.replace(' ', '_')[:50]
            
            parsed_url = urlparse(episode.audio_url)
            file_ext = Path(parsed_url.path).suffix or '.mp3'
            
            filename = f"{safe_title}{file_ext}"
            filepath = output_dir / filename
            
            if filepath.exists():
                logger.info(f"Audio file already exists: {filepath}")
                return filepath
            
            logger.info(f"Downloading: {episode.audio_url}")
            
            response = requests.get(episode.audio_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded audio: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading {episode.audio_url}: {e}")
            return None

    def _extract_podcast_name(self, feed) -> str:
        """Extract podcast name from feed metadata"""
        # Try multiple sources for podcast name
        podcast_name = (
            feed.feed.get('title', '') or
            feed.feed.get('subtitle', '') or
            'Unknown_Podcast'
        )
        
        # Sanitize for use as directory name
        safe_name = re.sub(r'[^\w\s-]', '', podcast_name)
        safe_name = re.sub(r'\s+', '_', safe_name.strip())
        return safe_name
    
    def _extract_episode_identifier(self, episode_title: str) -> Optional[str]:
        """Extract episode identifier from title (e.g., 'SN_1041' from 'SN 1041: Title')"""
        # Common patterns for episode identifiers
        patterns = [
            r'(SN\s*\d+)',  # Security Now: SN 1041
            r'(EP\s*\d+)',  # Episode EP 123
            r'(#\d+)',      # #123
            r'(\d+)',       # Just numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, episode_title, re.IGNORECASE)
            if match:
                identifier = match.group(1).replace(' ', '_').upper()
                return identifier
        
        # Fallback: use first few words of title
        words = episode_title.split()[:3]
        safe_words = [re.sub(r'[^\w]', '', word) for word in words if word]
        return '_'.join(safe_words) if safe_words else 'Unknown'
    
    def _extract_podcast_metadata(self, feed) -> Dict:
        """Extract comprehensive podcast-level metadata from feed"""
        metadata = {}
        
        # Basic feed info
        metadata['description'] = feed.feed.get('description', '')
        metadata['author'] = feed.feed.get('author', '')
        metadata['language'] = feed.feed.get('language', '')
        
        # Categories/tags
        categories = []
        if hasattr(feed.feed, 'tags'):
            categories.extend([tag.get('term', '') for tag in feed.feed.tags])
        if hasattr(feed.feed, 'category'):
            if isinstance(feed.feed.category, str):
                categories.append(feed.feed.category)
            elif isinstance(feed.feed.category, list):
                categories.extend(feed.feed.category)
        metadata['categories'] = [cat for cat in categories if cat]
        
        return metadata
    
    def _extract_episode_metadata(self, entry) -> Dict:
        """Extract comprehensive episode-level metadata from RSS entry"""
        metadata = {}
        
        # Episode numbering
        metadata['episode_number'] = getattr(entry, 'itunes_episode', None)
        metadata['season'] = getattr(entry, 'itunes_season', None)
        
        # Content info
        metadata['author'] = entry.get('author', '')
        metadata['summary'] = entry.get('summary', '')
        metadata['subtitle'] = getattr(entry, 'subtitle', None) or getattr(entry, 'itunes_subtitle', None)
        metadata['episode_type'] = getattr(entry, 'itunes_episodetype', 'full')
        
        # Explicit flag
        explicit_flag = getattr(entry, 'itunes_explicit', None)
        if explicit_flag:
            metadata['explicit'] = explicit_flag.lower() in ['true', 'yes', 'explicit']
        
        # Language
        metadata['language'] = entry.get('language', None)
        
        # Categories/tags for episode
        categories = []
        if hasattr(entry, 'tags'):
            categories.extend([tag.get('term', '') for tag in entry.tags])
        metadata['categories'] = [cat for cat in categories if cat]
        
        # Extract hosts from description or author field
        metadata['hosts'] = self._extract_hosts(entry)
        
        return metadata
    
    def _extract_hosts(self, entry) -> List[str]:
        """Attempt to extract host names from episode metadata"""
        hosts = []
        
        # Common patterns in TWiT shows
        author = entry.get('author', '')
        summary = entry.get('summary', '')
        description = entry.get('description', '')
        
        # For TWiT shows, hosts are often in the description
        text_to_search = f"{author} {summary} {description}".lower()
        
        # Known hosts for common shows
        known_hosts = {
            'leo laporte': 'Leo Laporte',
            'steve gibson': 'Steve Gibson',
            'jeff jarvis': 'Jeff Jarvis',
            'paris martineau': 'Paris Martineau',
            'ant pruitt': 'Ant Pruitt',
            'mikah sargent': 'Mikah Sargent'
        }
        
        for host_key, host_name in known_hosts.items():
            if host_key in text_to_search:
                hosts.append(host_name)
        
        # If no specific hosts found, use author field
        if not hosts and author and author != 'TWiT':
            hosts.append(author)
        
        return hosts