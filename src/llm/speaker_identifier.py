"""LLM-based speaker identification"""

import logging
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpeakerMapping:
    speaker_id: str
    identified_name: str
    confidence: float
    context: str


class SpeakerIdentifier:
    def __init__(self, provider: str = "openai", model: str = "gpt-4", 
                 api_key: str = "", temperature: float = 0.1, max_tokens: int = 4000):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def identify_speakers(self, diarized_segments: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
        """Identify speaker names from diarized transcript segments"""
        try:
            logger.info("Starting speaker identification with LLM")
            
            context_text = self._build_context_text(diarized_segments)
            
            speaker_mappings = self._query_llm_for_speakers(context_text)
            
            updated_segments = self._apply_speaker_mappings(diarized_segments, speaker_mappings)
            
            logger.info(f"Speaker identification complete: {len(speaker_mappings)} speakers identified")
            return updated_segments, speaker_mappings
            
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return diarized_segments, {}

    def _build_context_text(self, segments: List[Dict], max_segments: int = 50) -> str:
        """Build context text from first segments for speaker identification"""
        context_segments = segments[:max_segments]
        
        context_lines = []
        for seg in context_segments:
            start_time = self._format_timestamp(seg["start"])
            speaker = seg["speaker"]
            text = seg["text"].strip()
            
            if text:
                context_lines.append(f"[{start_time}] {speaker}: {text}")
        
        return "\n".join(context_lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _query_llm_for_speakers(self, context_text: str) -> Dict[str, str]:
        """Query LLM to identify speaker names from context"""
        prompt = f"""Analyze this podcast transcript and identify the real names of the speakers based on context clues like introductions, mentions of names, or other identifying information.

Return a JSON object mapping generic speaker IDs to identified names. Only include speakers you can confidently identify. If you cannot identify a speaker, do not include them in the mapping.

Format:
{{
    "SPEAKER_01": "John Doe",
    "SPEAKER_02": "Jane Smith"
}}

Transcript:
{context_text}

JSON Response:"""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.content[0].text
            
            return self._parse_speaker_response(content)
            
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return {}

    def _parse_speaker_response(self, response: str) -> Dict[str, str]:
        """Parse LLM response to extract speaker mappings"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                mappings = json.loads(json_str)
                
                validated_mappings = {}
                for speaker_id, name in mappings.items():
                    if isinstance(name, str) and name.strip() and "SPEAKER_" in speaker_id:
                        validated_mappings[speaker_id] = name.strip()
                
                return validated_mappings
            else:
                logger.warning("No valid JSON found in LLM response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {}

    def _apply_speaker_mappings(self, segments: List[Dict], 
                               mappings: Dict[str, str]) -> List[Dict]:
        """Apply speaker name mappings to transcript segments"""
        updated_segments = []
        
        for seg in segments:
            updated_seg = seg.copy()
            speaker_id = seg["speaker"]
            
            if speaker_id in mappings:
                updated_seg["speaker"] = mappings[speaker_id]
                updated_seg["original_speaker_id"] = speaker_id
            
            updated_segments.append(updated_seg)
        
        return updated_segments


class MockSpeakerIdentifier:
    """Mock speaker identifier for testing without API calls"""
    
    def identify_speakers(self, diarized_segments: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
        """Mock identification that creates simple mappings"""
        unique_speakers = set(seg["speaker"] for seg in diarized_segments)
        
        mappings = {}
        for i, speaker in enumerate(sorted(unique_speakers), 1):
            mappings[speaker] = f"Speaker_{i}"
        
        updated_segments = self._apply_speaker_mappings(diarized_segments, mappings)
        return updated_segments, mappings

    def _apply_speaker_mappings(self, segments: List[Dict], 
                               mappings: Dict[str, str]) -> List[Dict]:
        """Apply speaker name mappings to transcript segments"""
        updated_segments = []
        
        for seg in segments:
            updated_seg = seg.copy()
            speaker_id = seg["speaker"]
            
            if speaker_id in mappings:
                updated_seg["speaker"] = mappings[speaker_id]
                updated_seg["original_speaker_id"] = speaker_id
            
            updated_segments.append(updated_seg)
        
        return updated_segments