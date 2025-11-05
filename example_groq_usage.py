#!/usr/bin/env python3
"""
Example usage of Groq STT integration with WATUS
"""

import os
from groq_stt import GroqSTT

def main():
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Please set GROQ_API_KEY environment variable")
        print("   export GROQ_API_KEY='your_api_key_here'")
        return
    
    # Initialize Groq STT client
    model = "whisper-large-v3"  # or "whisper-large-v3-turbo" for faster processing
    client = GroqSTT(api_key, model)
    
    print(f"🎤 Initialized Groq STT with model: {model}")
    
    # Validate API key
    if client.validate_api_key(api_key):
        print("✅ API key is valid")
    else:
        print("❌ API key is invalid")
        return
    
    # Get available models
    print("\n📋 Available models:")
    for m in client.get_available_models():
        print(f"  - {m}")
    
    # Example transcription (you would need actual audio file)
    print("\n💡 To test with audio:")
    print("   1. Record some audio to a WAV file")
    print("   2. Use: client.transcribe_file('audio.wav', 'pl')")
    print("   3. Or convert numpy array: client.transcribe_numpy(audio_data, 16000, 'pl')")
    
    print("\n🚀 Integration with WATUS:")
    print("   1. Set STT_PROVIDER=groq in .env")
    print("   2. Add GROQ_API_KEY=your_key to .env")
    print("   3. Optionally set GROQ_MODEL=whisper-large-v3 in .env")

if __name__ == "__main__":
    main()
