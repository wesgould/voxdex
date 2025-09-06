"""Configuration management for AI Auto Transcripts"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
import os
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai or anthropic
    model: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000


@dataclass
class TranscriptionConfig:
    model_size: str = "base"  # tiny, base, small, medium, large
    language: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class DiarizationConfig:
    enabled: bool = True
    min_speakers: int = 1
    max_speakers: int = 10
    model: str = "pyannote/speaker-diarization-3.1"
    hf_token: Optional[str] = None


@dataclass
class OutputConfig:
    base_dir: str = "outputs"
    formats: List[str] = field(default_factory=lambda: ["txt", "json", "srt"])
    include_timestamps: bool = True


@dataclass
class PodcastFeed:
    name: str
    url: str
    max_episodes: Optional[int] = None
    enabled: bool = True


@dataclass
class Config:
    feeds: List[PodcastFeed] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class ConfigManager:
    def __init__(self, config_path: Union[str, Path]):
        load_dotenv()
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Config:
        if not self.config_path.exists():
            config = Config()
            self._save_default_config()
            return config

        with open(self.config_path, 'r') as f:
            if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return self._dict_to_config(data)

    def _dict_to_config(self, data: Dict) -> Config:
        feeds = [PodcastFeed(**feed) for feed in data.get('feeds', [])]
        
        llm_data = data.get('llm', {})
        llm_data['api_key'] = llm_data.get('api_key') or os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        diarization_data = data.get('diarization', {})
        diarization_data['hf_token'] = diarization_data.get('hf_token') or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        
        return Config(
            feeds=feeds,
            llm=LLMConfig(**llm_data),
            transcription=TranscriptionConfig(**data.get('transcription', {})),
            diarization=DiarizationConfig(**diarization_data),
            output=OutputConfig(**data.get('output', {}))
        )

    def _save_default_config(self):
        default_config = {
            'feeds': [
                {
                    'name': 'Example Podcast',
                    'url': 'https://example.com/feed.xml',
                    'max_episodes': 5,
                    'enabled': True
                }
            ],
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.1,
                'max_tokens': 4000
            },
            'transcription': {
                'model_size': 'base',
                'language': None,
                'device': 'auto'
            },
            'diarization': {
                'enabled': True,
                'min_speakers': 1,
                'max_speakers': 10,
                'model': 'pyannote/speaker-diarization-3.1'
            },
            'output': {
                'base_dir': 'outputs',
                'formats': ['txt', 'json', 'srt'],
                'include_timestamps': True
            }
        }

        with open(self.config_path, 'w') as f:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            else:
                json.dump(default_config, f, indent=2)

    def _validate_config(self):
        if not self.config.feeds:
            raise ValueError("No podcast feeds configured")
        
        if self.config.llm.provider not in ['openai', 'anthropic']:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")
        
        # Make API key optional - only warn if LLM features are used
        if not self.config.llm.api_key:
            import warnings
            warnings.warn(f"No API key configured for {self.config.llm.provider}. LLM speaker identification will be disabled.")

    def get_config(self) -> Config:
        return self.config