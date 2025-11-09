#!/usr/bin/env python3
"""
Groq Speech-to-Text integration for WATUS
"""

import asyncio
import base64
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, AsyncGenerator, Union
import numpy as np

import soundfile as sf
from groq import Groq
import logging

logger = logging.getLogger(__name__)


class GroqSTT:
    """Groq Speech-to-Text client with streaming and batch support"""
    
    def __init__(self, api_key: str, model: str = "whisper-large-v3"):
        """
        Initialize Groq STT client
        
        Args:
            api_key: Groq API key
            model: Model to use (whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en)
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"[GroqSTT] Initialized with model: {model}")
    
    def transcribe_file(self, audio_path: Union[str, Path], language: str = "pl") -> str:
        """
        Transcribe audio file using Groq API
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: "pl" for Polish)
            
        Returns:
            Transcribed text
        """
        logger.info(f"[GroqSTT] Transcribing file: {audio_path}")
        t0 = time.time()
        
        try:
            with open(audio_path, "rb") as file:
                # Transcribe the audio
                transcription = self.client.audio.transcriptions.create(
                    file=file,
                    model=self.model,
                    language=language,
                    response_format="json"
                )
                
            text = transcription.text.strip()
            duration_ms = int((time.time() - t0) * 1000)
            logger.info(f"[GroqSTT] Transcription completed in {duration_ms}ms: '{text[:50]}...'")
            
            return text
            
        except Exception as e:
            logger.error(f"[GroqSTT] Transcription failed: {e}")
            raise
    
    def transcribe_numpy(self, pcm_f32: np.ndarray, sample_rate: int = 16000, language: str = "pl") -> str:
        """
        Transcribe numpy array containing audio data
        
        Args:
            pcm_f32: Float32 numpy array with audio data
            sample_rate: Sample rate of the audio
            language: Language code (default: "pl" for Polish)
            
        Returns:
            Transcribed text
        """
        # Convert float32 to int16 for audio processing
        pcm_int16 = (pcm_f32 * 32767).astype(np.int16)
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Write audio to temporary file
            sf.write(temp_path, pcm_int16, sample_rate)
            
            # Transcribe using Groq API
            text = self.transcribe_file(temp_path, language)
            
            return text
            
        except Exception as e:
            logger.error(f"[GroqSTT] Numpy transcription failed: {e}")
            raise
            
        finally:
            # Clean up temporary file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
    
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None], language: str = "pl") -> AsyncGenerator[str, None]:
        """
        Transcribe audio stream in real-time using Groq API
        
        Args:
            audio_stream: Async generator yielding audio chunks
            language: Language code (default: "pl" for Polish)
            
        Yields:
            Transcribed text chunks as they become available
        """
        logger.info("[GroqSTT] Starting streaming transcription")
        
        # Collect audio chunks
        audio_chunks = []
        async for chunk in audio_stream:
            audio_chunks.append(chunk)
        
        if not audio_chunks:
            return
        
        # Combine all chunks
        full_audio = b''.join(audio_chunks)
        
        # Create temporary file with audio data
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(full_audio)
        
        try:
            # Transcribe using Groq API
            text = self.transcribe_file(temp_path, language)
            yield text
            
        except Exception as e:
            logger.error(f"[GroqSTT] Streaming transcription failed: {e}")
            raise
            
        finally:
            # Clean up temporary file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """
        Validate Groq API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            client = Groq(api_key=api_key)
            # Try to get models list to validate key
            client.models.list()
            return True
        except Exception as e:
            logger.error(f"[GroqSTT] API key validation failed: {e}")
            return False
    
    @staticmethod
    def get_available_models() -> list:
        """Get list of available Groq Whisper models"""
        return [
            "whisper-large-v3",      # Best quality
            "whisper-large-v3-turbo", # Fast variant  
            "distil-whisper-large-v3-en"  # English-only, fastest
        ]


# Utility functions for audio format conversion
def convert_float32_to_wav_bytes(pcm_f32: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Convert float32 audio to WAV bytes
    
    Args:
        pcm_f32: Float32 numpy array with audio data
        sample_rate: Sample rate of the audio
        
    Returns:
        WAV bytes
    """
    # Convert float32 to int16
    pcm_int16 = (pcm_f32 * 32767).astype(np.int16)
    
    # Write to temporary bytes buffer
    from io import BytesIO
    buffer = BytesIO()
    sf.write(buffer, pcm_int16, sample_rate)
    
    return buffer.getvalue()


def get_audio_info(audio_data: Union[np.ndarray, str, Path]) -> dict:
    """
    Get audio file information
    
    Args:
        audio_data: Audio data (numpy array) or path to audio file
        
    Returns:
        Dictionary with audio information
    """
    if isinstance(audio_data, (str, Path)):
        info = sf.info(audio_data)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames
        }
    else:
        return {
            "shape": audio_data.shape,
            "dtype": str(audio_data.dtype),
            "sample_rate": "unknown"  # Would need to be passed separately
        }


if __name__ == "__main__":
    # Test basic functionality
    import os
    
    # Example usage
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set GROQ_API_KEY environment variable")
        exit(1)
    
    # Initialize client
    client = GroqSTT(api_key)
    
    # Validate API key
    if client.validate_api_key(api_key):
        print("✅ API key is valid")
    else:
        print("❌ API key is invalid")
        exit(1)
    
    print(f"Available models: {client.get_available_models()}")
