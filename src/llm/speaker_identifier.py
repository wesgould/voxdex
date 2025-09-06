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
        # For GPT-5 series compatibility (nano, mini, etc.)
        self.use_gpt5_series = model.startswith("gpt-5")
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

    def identify_speakers(self, diarized_segments: List[Dict], metadata: Optional[Dict] = None) -> Tuple[List[Dict], Dict[str, str]]:
        """Identify speaker names from diarized transcript segments"""
        try:
            logger.info("Starting speaker identification with LLM")
            
            context_text = self._build_context_text(diarized_segments)
            
            speaker_mappings = self._query_llm_for_speakers(context_text, metadata)
            
            updated_segments = self._apply_speaker_mappings(diarized_segments, speaker_mappings)
            
            logger.info(f"Speaker identification complete: {len(speaker_mappings)} speakers identified")
            return updated_segments, speaker_mappings
            
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return diarized_segments, {}

    def _build_context_text(self, segments: List[Dict]) -> str:
        """Build context text from transcript for speaker identification"""
        context_lines = []
        
        # For large transcripts (>6 speakers), use a sampling approach
        unique_speakers = set(seg["speaker"] for seg in segments)
        
        if len(unique_speakers) > 6:
            # For very complex episodes, use more aggressive sampling
            max_segments = 150 if len(unique_speakers) > 10 else 200
            
            # For Intelligent Machines-style shows: interview first, then hosts discussion
            # Sample from both interview (first ~30 min) and main show (after interview)
            total_duration = segments[-1]["start"] if segments else 1800  # Default 30min
            interview_cutoff = min(1800, total_duration * 0.4)  # First 30min or 40% of show
            
            interview_segments = [s for s in segments if s["start"] <= interview_cutoff]
            main_show_segments = [s for s in segments if s["start"] > interview_cutoff]
            
            sampled_segments = []
            
            # From interview: intro + key transitions
            if interview_segments:
                sampled_segments.extend(interview_segments[:30])  # Show intro + interview start
                
                # Add interview speaker transitions
                prev_speaker = None
                for seg in interview_segments[30:]:
                    if seg["speaker"] != prev_speaker:
                        seg_idx = interview_segments.index(seg)
                        sampled_segments.extend(interview_segments[max(0, seg_idx-1):seg_idx+2])
                    prev_speaker = seg["speaker"]
            
            # From main show: transitions + substantial segments
            if main_show_segments:
                sampled_segments.extend(main_show_segments[:20])  # Transition to main show
                
                # Add main show speaker transitions
                prev_speaker = None
                transition_count = 0
                for seg in main_show_segments[20:]:
                    if seg["speaker"] != prev_speaker and transition_count < 15:
                        seg_idx = main_show_segments.index(seg)
                        sampled_segments.extend(main_show_segments[max(0, seg_idx-1):seg_idx+2])
                        transition_count += 1
                    prev_speaker = seg["speaker"]
            
            # Fallback: if no clear interview/main split, use original strategy
            if not interview_segments or not main_show_segments:
                sampled_segments = segments[:50]
                
                speakers_seen = set()
                for seg in segments:
                    if seg["speaker"] not in speakers_seen:
                        speakers_seen.add(seg["speaker"])
                        seg_idx = segments.index(seg)
                        sampled_segments.extend(segments[max(0, seg_idx-1):seg_idx+3])
            
            # Remove duplicates while preserving order
            seen = set()
            segments_to_use = []
            for seg in sampled_segments:
                seg_id = (seg["start"], seg["speaker"], seg["text"])
                if seg_id not in seen:
                    segments_to_use.append(seg)
                    seen.add(seg_id)
            
            segments_to_use = segments_to_use[:max_segments]
        else:
            segments_to_use = segments
        
        for seg in segments_to_use:
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

    def _query_llm_for_speakers(self, context_text: str, metadata: Optional[Dict] = None) -> Dict[str, str]:
        """Query LLM to identify speaker names from context"""
        
        # Build metadata context
        metadata_context = ""
        if metadata:
            metadata_parts = []
            
            # Episode info
            episode_info = metadata.get("episode", {})
            if episode_info.get("title"):
                metadata_parts.append(f"Episode: {episode_info['title']}")
            if episode_info.get("hosts"):
                hosts_list = ", ".join(episode_info["hosts"])
                metadata_parts.append(f"Known Hosts: {hosts_list}")
            
            # Extract guest information from description/summary
            description = episode_info.get("description", "") + " " + episode_info.get("summary", "")
            if description:
                # Look for guest patterns in HTML content
                import re
                guest_patterns = [
                    r'<strong>Guests?:</strong>\s*(.+?)(?:</p>|<p>)',
                    r'<strong>Guest:</strong>\s*(.+?)(?:</p>|<p>)', 
                    r'<strong>Co-Host:</strong>\s*(.+?)(?:</p>|<p>)',
                ]
                
                all_guests = []
                for pattern in guest_patterns:
                    for match in re.finditer(pattern, description, re.IGNORECASE | re.DOTALL):
                        guests_text = match.group(1).strip()
                        # Clean up HTML tags and extract names
                        guests_clean = re.sub(r'<[^>]+>', '', guests_text)
                        guests_clean = re.sub(r'\s+', ' ', guests_clean).strip()
                        if guests_clean and guests_clean not in ['', 'None']:
                            all_guests.append(guests_clean)
                
                if all_guests:
                    metadata_parts.append(f"Known Guests: {', '.join(all_guests)}")
            
            # Podcast info
            podcast_info = metadata.get("podcast", {})
            if podcast_info.get("name"):
                metadata_parts.append(f"Podcast: {podcast_info['name']}")
                
            if metadata_parts:
                metadata_context = f"""

Episode Metadata:
{chr(10).join(metadata_parts)}
"""
        
        prompt = f"""You are a speaker identification expert. Analyze this podcast transcript and identify speakers.

TASK: Map each SPEAKER_ID to a real name using context clues and metadata.

RULES:
1. Use introductions, greetings, and names mentioned in the transcript
2. Match voices to the known hosts/guests from metadata
3. Label unidentifiable speakers as "Unknown" or "Voiceover" 
4. Return ONLY valid JSON - no explanations or extra text

OUTPUT FORMAT (REQUIRED):
Return exactly this JSON structure:
{{
  "SPEAKER_00": "Speaker Name",
  "SPEAKER_01": "Another Name",
  "SPEAKER_02": "Unknown"
}}
{metadata_context}
TRANSCRIPT:
{context_text}

RESPOND WITH ONLY THE JSON MAPPING:"""

        # Debug: log prompt size and unique speakers
        unique_speakers_in_context = len(set(re.findall(r'SPEAKER_\d+', context_text)))
        logger.debug(f"Sending prompt with {len(prompt)} chars, {len(context_text.splitlines())} context lines, {unique_speakers_in_context} unique speakers")

        try:
            if self.provider == "openai":
                # Handle GPT-5 series parameters (nano, mini, etc.)
                if self.use_gpt5_series:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=self.max_tokens
                    )
                else:
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
            # Clean up response - remove common formatting issues
            cleaned_response = response.strip()
            
            # Try to find JSON block
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces handling
                r'\{.*?\}',  # Simple pattern
            ]
            
            json_str = None
            for pattern in json_patterns:
                json_match = re.search(pattern, cleaned_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    break
            
            if not json_str:
                # Try parsing entire response as JSON if it looks like JSON
                if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                    json_str = cleaned_response
            
            if json_str:
                # Clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                
                mappings = json.loads(json_str)
                
                validated_mappings = {}
                for speaker_id, name in mappings.items():
                    if isinstance(name, str) and name.strip() and "SPEAKER_" in speaker_id:
                        validated_mappings[speaker_id] = name.strip()
                
                logger.info(f"Successfully parsed {len(validated_mappings)} speaker mappings")
                return validated_mappings
            else:
                logger.warning(f"No valid JSON found in LLM response: {response[:500]}...")
                logger.debug(f"Full LLM response: {response}")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            return {}

    def _apply_speaker_mappings(self, segments: List[Dict], 
                               mappings: Dict[str, str]) -> List[Dict]:
        """Apply speaker name mappings to transcript segments using regex replacement"""
        import re
        
        updated_segments = []
        
        for seg in segments:
            updated_seg = seg.copy()
            speaker_id = seg["speaker"]
            text = seg["text"]
            
            # Use regex to replace speaker IDs in the text as well
            updated_text = text
            for old_speaker, new_name in mappings.items():
                # Replace speaker mentions in the text content
                pattern = re.compile(re.escape(old_speaker), re.IGNORECASE)
                updated_text = pattern.sub(new_name, updated_text)
            
            # Update the speaker field if there's a mapping
            if speaker_id in mappings:
                updated_seg["speaker"] = mappings[speaker_id]
                updated_seg["original_speaker_id"] = speaker_id
            
            updated_seg["text"] = updated_text
            updated_segments.append(updated_seg)
        
        return updated_segments


class MockSpeakerIdentifier:
    """Mock speaker identifier for testing without API calls"""
    
    def identify_speakers(self, diarized_segments: List[Dict], metadata: Optional[Dict] = None) -> Tuple[List[Dict], Dict[str, str]]:
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