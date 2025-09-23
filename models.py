# models.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LeaderBlock:
    session_id: str
    group_id: str
    speaker_id: str
    text_full: str
    category: str
    reply_hint: bool
    ts_start: float
    ts_end: float
    verify: Optional[Dict[str, Any]] = None

@dataclass
class TTSCommand:
    speak: str
    rate: int = 180
    volume: float = 0.95
    voice_hint: Optional[str] = None
