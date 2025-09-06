#!/usr/bin/env python3
"""
Test script for LLM speaker identification enhancement
Uses the most recent Intelligent Machines podcast transcript
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.llm.speaker_identifier import SpeakerIdentifier, MockSpeakerIdentifier


def load_diarized_transcript(json_path: str):
    """Load diarized transcript from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['segments']


def print_first_lines(segments, num_lines=10):
    """Print first N lines of transcript"""
    print(f"\n=== First {num_lines} lines of transcript ===")
    for i, seg in enumerate(segments[:num_lines]):
        start_time = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
        speaker = seg['speaker']
        text = seg['text'].strip()
        print(f"[{start_time}] {speaker}: {text}")


def test_context_building(segments):
    """Test the context building function"""
    print("\n=== Testing Context Building (Entire Transcript) ===")
    
    # Build context text from entire transcript
    context_lines = []
    
    for seg in segments:
        start_time = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
        speaker = seg['speaker']
        text = seg['text'].strip()
        
        if text:
            context_lines.append(f"[{start_time}] {speaker}: {text}")
    
    context_text = "\n".join(context_lines)
    lines = context_text.split('\n')
    
    print(f"Context text generated: {len(lines)} lines (entire transcript)")
    print(f"Total characters: {len(context_text):,}")
    print("First 5 lines:")
    for line in lines[:5]:
        print(f"  {line}")
    print("...")
    print("Last 5 lines:")
    for line in lines[-5:]:
        print(f"  {line}")


def test_mock_identification(segments):
    """Test mock speaker identification"""
    print("\n=== Testing Mock Speaker Identification ===")
    
    mock_identifier = MockSpeakerIdentifier()
    enhanced_segments, speaker_mappings = mock_identifier.identify_speakers(segments)
    
    print(f"Speaker mappings found: {speaker_mappings}")
    print(f"Original segments: {len(segments)}")
    print(f"Enhanced segments: {len(enhanced_segments)}")
    
    # Show a few enhanced segments
    print("\nFirst 3 enhanced segments:")
    for i, seg in enumerate(enhanced_segments[:3]):
        print(f"  Segment {i+1}: {seg['speaker']} -> {seg.get('original_speaker_id', 'N/A')}")
        print(f"    Text: {seg['text'][:100]}...")


def test_real_llm_identification(segments):
    """Test real LLM speaker identification with entire transcript (requires API key)"""
    print("\n=== Testing Real LLM Speaker Identification (Full Transcript) ===")
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("No OPENAI_API_KEY found in environment - skipping real LLM test")
        print("Set OPENAI_API_KEY to test with actual LLM")
        return
    
    # Build entire transcript context
    context_lines = []
    for seg in segments:
        start_time = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
        speaker = seg['speaker']
        text = seg['text'].strip()
        
        if text:
            context_lines.append(f"[{start_time}] {speaker}: {text}")
    
    full_context = "\n".join(context_lines)
    
    print(f"Sending entire transcript to GPT-5-nano ({len(context_lines)} lines, {len(full_context):,} characters)")
    
    # Create custom prompt for entire transcript
    prompt = f"""Here is the complete transcript of a podcast. Each line starts with a timestamp and SPEAKER_ID.
Please output a JSON mapping of SPEAKER_ID → real name, using context clues (introductions, greetings, etc.).
If the speaker cannot be identified, label them "Unknown" or "Voiceover" as appropriate.

Transcript:
{full_context}

JSON Response:"""
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000
        )
        
        content = response.choices[0].message.content
        print(f"LLM Response: {content}")
        
        # Parse the response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            speaker_mappings = json.loads(json_str)
            
            print(f"\nParsed Speaker mappings: {speaker_mappings}")
            
            # Apply mappings and show results
            enhanced_count = 0
            for seg in segments[:5]:  # Show first 5
                speaker_id = seg["speaker"]
                if speaker_id in speaker_mappings:
                    print(f"  {speaker_id} -> {speaker_mappings[speaker_id]}")
                    enhanced_count += 1
                else:
                    print(f"  {speaker_id} -> (unchanged)")
                    
            print(f"\nTotal speakers mapped: {len(speaker_mappings)}")
            
        else:
            print("No valid JSON found in LLM response")
            
    except Exception as e:
        print(f"Error testing real LLM: {e}")


def main():
    # Find the most recent Intelligent Machines transcript
    im_dir = Path("output/Intelligent_Machines_Audio")
    
    if not im_dir.exists():
        print(f"Error: {im_dir} does not exist")
        print("Please run the main pipeline first to generate transcripts")
        return
    
    # Get all episode directories
    episode_dirs = [d for d in im_dir.iterdir() if d.is_dir()]
    if not episode_dirs:
        print("No Intelligent Machines episodes found")
        return
    
    # Use the most recent one (IM_835 based on the directory listing)
    latest_episode = max(episode_dirs, key=lambda x: x.name)
    
    # Find the diarized JSON file
    diarized_file = None
    for file in latest_episode.iterdir():
        if file.name.endswith('_diarized.json'):
            diarized_file = file
            break
    
    if not diarized_file:
        print(f"No diarized JSON file found in {latest_episode}")
        return
    
    print(f"Testing with: {diarized_file}")
    
    # Load the transcript
    try:
        segments = load_diarized_transcript(diarized_file)
        print(f"Loaded {len(segments)} transcript segments")
        
        # Print some basic info
        unique_speakers = set(seg['speaker'] for seg in segments)
        print(f"Unique speakers found: {unique_speakers}")
        
        # Show first few lines
        print_first_lines(segments)
        
        # Test context building
        test_context_building(segments)
        
        # Test mock identification
        test_mock_identification(segments)
        
        # Test real LLM identification
        test_real_llm_identification(segments)
        
    except Exception as e:
        print(f"Error loading transcript: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()