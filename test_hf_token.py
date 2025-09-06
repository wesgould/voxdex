#!/usr/bin/env python3
"""Test HuggingFace token access to pyannote model"""

def test_hf_token():
    """Test if HF token can access pyannote diarization model"""
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("❌ No HF_TOKEN found in .env file")
        print("Please add your token to .env file: HF_TOKEN=your_token_here")
        return
    model_name = "pyannote/speaker-diarization-3.1"
    
    print("Testing HuggingFace token access...")
    print(f"Token: {hf_token[:10]}...")
    print(f"Model: {model_name}")
    
    try:
        from pyannote.audio import Pipeline
        
        print("Attempting to load diarization pipeline...")
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        
        print("✅ SUCCESS! Token works and model loaded successfully")
        print("You can now run: python main.py")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print("\nNext steps:")
        print("1. Visit: https://hf.co/pyannote/speaker-diarization-3.1")
        print("2. Click 'Agree and access repository'") 
        print("3. Make sure you're logged in with your new account")
        print("4. Try this test again")

if __name__ == "__main__":
    test_hf_token()