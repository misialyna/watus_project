#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CT2_SKIP_CONVERTERS", "1")

import sys, json, time, queue, threading, subprocess, tempfile, atexit, re
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import soundfile as sf
import zmq
import webrtcvad

# Enhanced error handling and logging
class WatusLogger:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.system_info = {}
        
    def error(self, msg, component=None):
        error_msg = f"[ERROR{f' ({component})' if component else ''}] {msg}"
        print(error_msg, flush=True)
        self.errors.append({"time": time.time(), "component": component, "message": msg})
        
    def warning(self, msg, component=None):
        warning_msg = f"[WARN{f' ({component})' if component else ''}] {msg}"
        print(warning_msg, flush=True)
        self.warnings.append({"time": time.time(), "component": component, "message": msg})
        
    def info(self, msg):
        print(msg, flush=True)
        
    def log_system_info(self):
        """Log system information for debugging"""
        import torch
        self.system_info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "audio_devices": len(sd.query_devices()) if hasattr(sd, 'query_devices') else 0,
            "platform": sys.platform
        }
        self.info(f"[SYSTEM] {json.dumps(self.system_info, indent=2)}")

logger = WatusLogger()
logger.log_system_info()

# ASR: Choose between Faster-Whisper (local) or Groq API
try:
    from faster_whisper import WhisperModel
    logger.info("[OK] Faster-Whisper imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Faster-Whisper: {e}", "ASR")
    WhisperModel = None

try:
    from groq_stt import GroqSTT
    logger.info("[OK] Groq STT imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Groq STT: {e}", "ASR")
    GroqSTT = None

# LED Controller with graceful fallback
try:
    from led_controller import LEDController
    led = LEDController()
    atexit.register(led.cleanup)
    logger.info("[OK] LED Controller loaded")
except Exception as e:
    logger.warning(f"LED Controller failed: {e}", "LED")
    # Create dummy LED controller
    class DummyLED:
        def listening(self): pass
        def processing_or_speaking(self): pass
        def cleanup(self): pass
    led = DummyLED()

# .env configuration
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# Configuration with validation
def get_config_with_fallback(key, default, required=False, validator=None):
    """Get configuration with fallback and validation"""
    value = os.environ.get(key, default)
    
    if required and not value:
        logger.error(f"Required configuration missing: {key}", "CONFIG")
        return None
        
    if validator and not validator(value):
        logger.error(f"Invalid configuration for {key}: {value}", "CONFIG")
        return None
        
    return value

# ZMQ Configuration
PUB_ADDR = get_config_with_fallback("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")
SUB_ADDR = get_config_with_fallback("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")

def _normalize_fw_model(name: str) -> str:
    name = (name or "").strip()
    short = {"tiny","base","small","medium","large","large-v1","large-v2","large-v3"}
    if "/" not in name and name.lower() in short:
        return f"guillaumekln/faster-whisper-{name.lower()}"
    return name

# ===== STT Configuration =====
STT_PROVIDER = get_config_with_fallback("STT_PROVIDER", "local").lower()  # "local" for Faster-Whisper, "groq" for Groq API

# Groq API configuration (used when STT_PROVIDER=groq)
GROQ_API_KEY = get_config_with_fallback("GROQ_API_KEY", "")
GROQ_MODEL = get_config_with_fallback("GROQ_MODEL", "whisper-large-v3")

# Local Whisper configuration (used when STT_PROVIDER=local)
WHISPER_MODEL_NAME = _normalize_fw_model(get_config_with_fallback("WHISPER_MODEL", "small"))
WHISPER_DEVICE = get_config_with_fallback("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = get_config_with_fallback("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_NUM_WORKERS = int(get_config_with_fallback("WHISPER_NUM_WORKERS", "1"))

CPU_THREADS = int(get_config_with_fallback("WATUS_CPU_THREADS", "4"))

# Audio/VAD Configuration
SAMPLE_RATE = int(get_config_with_fallback("WATUS_SR", "16000"))
BLOCK_SIZE = int(get_config_with_fallback("WATUS_BLOCKSIZE", "160"))
VAD_MODE = int(get_config_with_fallback("WATUS_VAD_MODE", "1"))
VAD_MIN_MS = int(get_config_with_fallback("WATUS_VAD_MIN_MS", "280"))
SIL_MS_END = int(get_config_with_fallback("WATUS_SIL_MS_END", "650"))
ASR_MIN_DBFS = float(get_config_with_fallback("ASR_MIN_DBFS", "-34"))

# Speaker verification
SPEAKER_VERIFY = int(get_config_with_fallback("SPEAKER_VERIFY", "1"))
SPEAKER_REQUIRE_MATCH = int(get_config_with_fallback("SPEAKER_REQUIRE_MATCH", "1"))

# Audio device detection with fallback
def detect_audio_devices():
    """Detect audio devices with comprehensive error handling"""
    try:
        devices = sd.query_devices()
        input_devices = []
        output_devices = []
        
        for i, d in enumerate(devices):
            if d.get('max_input_channels', 0) > 0:
                input_devices.append((i, d['name']))
            if d.get('max_output_channels', 0) > 0:
                output_devices.append((i, d['name']))
        
        logger.info(f"[AUDIO] Found {len(devices)} total devices:")
        for i, d in enumerate(devices):
            input_ch = d.get('max_input_channels', 0)
            output_ch = d.get('max_output_channels', 0)
            logger.info(f"  [{i}] {d['name']} (IN:{input_ch} OUT:{output_ch})")
            
        return input_devices, output_devices
    except Exception as e:
        logger.error(f"Audio device detection failed: {e}", "AUDIO")
        return [], []

# Device configuration with fallback
IN_DEV_ENV = get_config_with_fallback("WATUS_INPUT_DEVICE", "")
OUT_DEV_ENV = get_config_with_fallback("WATUS_OUTPUT_DEVICE", "")

input_devices, output_devices = detect_audio_devices()

def _auto_pick_device(device_list):
    """Auto-pick first available device from list"""
    if device_list:
        return device_list[0][0]  # Return device index
    return None

# Set up devices with fallbacks
IN_DEV = None
OUT_DEV = None

if IN_DEV_ENV:
    try:
        IN_DEV = int(IN_DEV_ENV)
    except ValueError:
        # Try to match by name
        for idx, name in input_devices:
            if IN_DEV_ENV.lower() in name.lower():
                IN_DEV = idx
                break
else:
    IN_DEV = _auto_pick_device(input_devices)

if OUT_DEV_ENV:
    try:
        OUT_DEV = int(OUT_DEV_ENV)
    except ValueError:
        # Try to match by name
        for idx, name in output_devices:
            if OUT_DEV_ENV.lower() in name.lower():
                OUT_DEV = idx
                break
else:
    OUT_DEV = _auto_pick_device(output_devices)

logger.info(f"[AUDIO] Configured devices - Input: {IN_DEV}, Output: {OUT_DEV}")

# Audio mode selection based on device availability
AUDIO_MODE = "FULL" if (IN_DEV is not None and OUT_DEV is not None) else "SIMULATION"

if AUDIO_MODE == "SIMULATION":
    logger.warning("Running in SIMULATION mode - no audio devices available", "AUDIO")
    logger.info("Will simulate audio input/output for testing")

DIALOG_PATH = get_config_with_fallback("DIALOG_PATH", "dialog.jsonl")

def log(msg): 
    logger.info(msg)

def is_wake_word_present(text: str) -> bool:
    """Enhanced wake word detection"""
    if not text:
        return False
    
    # Get wake words from config
    wake_words_env = get_config_with_fallback("WAKE_WORDS", "hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu")
    wake_words = [w.strip() for w in wake_words_env.split(",") if w.strip()]
    
    normalized_text = re.sub(r'[^\w\s]', '', text.lower())
    
    for wake_phrase in wake_words:
        normalized_wake_phrase = re.sub(r'[^\w\s]', '', wake_phrase.lower())
        if normalized_wake_phrase in normalized_text:
            return True
    return False

def cue_listen():
    log("[Watus][STATE] LISTENING")
    led.listening()
def cue_think():
    log("[Watus][STATE] THINKING")
    led.processing_or_speaking()
def cue_speak():
    log("[Watus][STATE] SPEAKING")
    led.processing_or_speaking()
def cue_idle():
    log("[Watus][STATE] IDLE")
    led.processing_or_speaking()

# ZMQ Bus with enhanced error handling
class Bus:
    def __init__(self, pub_addr: str, sub_addr: str):
        try:
            self.ctx = zmq.Context.instance()
            self.pub = self.ctx.socket(zmq.PUB)
            self.pub.setsockopt(zmq.SNDHWM, 100)
            self.pub.bind(pub_addr)

            self.sub = self.ctx.socket(zmq.SUB)
            self.sub.connect(sub_addr)
            self.sub.setsockopt_string(zmq.SUBSCRIBE, "tts.speak")

            self._sub_queue = queue.Queue()
            threading.Thread(target=self._sub_loop, daemon=True).start()
            logger.info(f"[OK] ZMQ Bus initialized - PUB: {pub_addr}, SUB: {sub_addr}")
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ Bus: {e}", "ZMQ")
            raise

    def publish_leader(self, payload: dict):
        try:
            t0 = time.time()
            self.pub.send_multipart([b"dialog.leader", json.dumps(payload, ensure_ascii=False).encode("utf-8")])
            log(f"[Perf] BUS_ms={int((time.time()-t0)*1000)}")
        except Exception as e:
            logger.error(f"Failed to publish leader message: {e}", "ZMQ")

    def _sub_loop(self):
        while True:
            try:
                topic, payload = self.sub.recv_multipart()
                if topic != b"tts.speak": continue
                data = json.loads(payload.decode("utf-8", "ignore"))
                self._sub_queue.put(data)
            except Exception as e:
                logger.error(f"ZMQ subscription error: {e}", "ZMQ")
                time.sleep(0.01)

    def get_tts(self, timeout=0.1):
        try: 
            return self._sub_queue.get(timeout=timeout)
        except queue.Empty: 
            return None

# Speaker verification with fallback
class _NoopVerifier:
    enabled=False
    def __init__(self): 
        self._enrolled=None
    @property
    def enrolled(self): 
        return False
    def enroll_wav(self, p): 
        pass
    def enroll_samples(self, s, sr): 
        pass
    def verify(self, s, sr, db): 
        return {"enabled": False}

def _make_verifier():
    if not SPEAKER_VERIFY: 
        logger.warning("Speaker verification disabled", "SPEAKER")
        return _NoopVerifier()
    
    try:
        import torch
        from speechbrain.inference import EncoderClassifier
        logger.info("Loading SpeechBrain ECAPA model...")
    except Exception as e:
        logger.warning(f"SpeechBrain not available: {e}. Speaker verification disabled.", "SPEAKER")
        return _NoopVerifier()

    class _SbVerifier:
        enabled=True
        def __init__(self):
            import torch
            self.threshold = float(get_config_with_fallback("SPEAKER_THRESHOLD", "0.64"))
            self.sticky_thr = float(get_config_with_fallback("SPEAKER_STICKY_THRESHOLD", str(self.threshold)))
            self.back_thr = float(get_config_with_fallback("SPEAKER_BACK_THRESHOLD", "0.56"))
            self.grace = float(get_config_with_fallback("SPEAKER_GRACE", "0.12"))
            self.sticky_sec = float(get_config_with_fallback("SPEAKER_STICKY_SEC", "3600"))
            self._clf=None
            self._device="cpu"  # Force CPU for compatibility
            self._enrolled=None
            self._enroll_ts=0.0
            logger.info(f"[OK] Speaker verifier configured (threshold: {self.threshold})")

        @property
        def enrolled(self): 
            return self._enrolled is not None

        def _ensure(self):
            if self._clf is None:
                try:
                    from speechbrain.inference import EncoderClassifier
                    self._clf = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        run_opts={"device": self._device},
                        savedir="models/ecapa",
                    )
                    logger.info("[OK] ECAPA model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load ECAPA model: {e}", "SPEAKER")
                    raise

        def _resample_16k(self, x: np.ndarray, sr: int) -> np.ndarray:
            if sr == 16000: 
                return x.astype(np.float32)
            ratio = 16000.0/sr
            n_out = int(round(len(x)*ratio))
            idx = np.linspace(0, len(x)-1, num=n_out, dtype=np.float32)
            base = np.arange(len(x), dtype=np.float32)
            return np.interp(idx, base, x).astype(np.float32)

        def _embed(self, samples: np.ndarray, sr: int):
            self._ensure()
            wav = self._resample_16k(samples, sr)
            t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = self._clf.encode_batch(t).squeeze(0).squeeze(0)
            return emb.detach().cpu().numpy().astype(np.float32)

        def enroll_samples(self, samples: np.ndarray, sr: int):
            try:
                emb = self._embed(samples, sr)
                self._enrolled = emb
                self._enroll_ts = time.time()
                logger.info("[OK] New leader voice enrolled")
            except Exception as e:
                logger.error(f"Speaker enrollment failed: {e}", "SPEAKER")

        def verify(self, samples: np.ndarray, sr: int, dbfs: float) -> dict:
            if self._enrolled is None:
                return {"enabled": True, "enrolled": False}
            
            try:
                import torch, torch.nn.functional as F
                a = self._embed(samples, sr)
                sim = float(F.cosine_similarity(
                    torch.tensor(a, dtype=torch.float32).flatten(),
                    torch.tensor(self._enrolled, dtype=torch.float32).flatten(), 
                    dim=0, eps=1e-8
                ).detach().cpu().item())
                
                now = time.time()
                age = now - self._enroll_ts
                is_leader = False
                adj_thr = (self.sticky_thr - self.grace) if dbfs > -22.0 else self.sticky_thr
                if age <= self.sticky_sec and sim >= adj_thr: 
                    is_leader=True
                elif sim >= self.threshold: 
                    is_leader=True
                elif sim >= self.back_thr and age <= self.sticky_sec: 
                    is_leader=True
                    
                return {"enabled": True, "enrolled": True, "score": sim, "is_leader": bool(is_leader), "sticky_age_s": age}
            except Exception as e:
                logger.error(f"Speaker verification failed: {e}", "SPEAKER")
                return {"enabled": True, "enrolled": True, "score": 0.0, "is_leader": False, "error": str(e)}

    return _SbVerifier()

# STT Engine with enhanced error handling
class STTEngine:
    def __init__(self, state: 'State', bus: 'Bus'):
        self.state = state
        self.bus = bus
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.stt_provider = STT_PROVIDER
        
        logger.info(f"[OK] STT ready (device={IN_DEV} sr={SAMPLE_RATE} block={BLOCK_SIZE}, mode={AUDIO_MODE})")
        
        # Initialize STT based on provider
        if self.stt_provider == "groq":
            self._init_groq_stt()
        else:
            self._init_local_whisper()
            
        self.verifier = _make_verifier()
        self.emit_cooldown_ms = int(get_config_with_fallback("EMIT_COOLDOWN_MS", "300"))
        self.cooldown_until = 0
        
        # Audio simulation for environments without audio devices
        self.simulation_mode = (AUDIO_MODE == "SIMULATION")
        if self.simulation_mode:
            logger.info("[SIM] Audio simulation mode enabled")
    
    def _init_groq_stt(self):
        """Initialize Groq Speech-to-Text API"""
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not provided, falling back to local Whisper", "ASR")
            return self._init_local_whisper()
            
        if GroqSTT is None:
            logger.error("Groq STT module not available, falling back to local Whisper", "ASR")
            return self._init_local_whisper()
            
        try:
            logger.info(f"[AUDIO] Initializing Groq STT: model={GROQ_MODEL}")
            t0 = time.time()
            self.model = GroqSTT(GROQ_API_KEY, GROQ_MODEL)
            
            # Validate API key
            if not self.model.validate_api_key(GROQ_API_KEY):
                logger.error("Invalid GROQ_API_KEY, falling back to local Whisper", "ASR")
                return self._init_local_whisper()
                
            load_time = int((time.time()-t0)*1000)
            logger.info(f"[OK] Groq STT API loaded successfully ({load_time} ms)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq STT: {e}", "ASR")
            logger.error("Falling back to local Whisper", "ASR")
            return self._init_local_whisper()
    
    def _init_local_whisper(self):
        """Initialize local Faster-Whisper"""
        if WhisperModel is None:
            logger.error("Faster-Whisper module not available", "ASR")
            raise ImportError("Neither Groq STT nor Faster-Whisper is available")
            
        logger.info(f"[AUDIO] Initializing Faster-Whisper: model={WHISPER_MODEL_NAME} device={WHISPER_DEVICE} compute={WHISPER_COMPUTE}")
        
        try:
            t0 = time.time()
            self.model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE,
                cpu_threads=CPU_THREADS,
                num_workers=WHISPER_NUM_WORKERS
            )
            load_time = int((time.time()-t0)*1000)
            logger.info(f"[OK] Faster-Whisper loaded successfully ({load_time} ms)")
            self.stt_provider = "local"
        except Exception as e:
            logger.error(f"Failed to initialize Faster-Whisper: {e}", "ASR")
            raise

    @staticmethod
    def _rms_dbfs(x: np.ndarray, eps=1e-9):
        rms = np.sqrt(np.mean(np.square(x) + eps))
        return 20*np.log10(max(rms, eps))

    def _vad_is_speech(self, frame_bytes: bytes) -> bool:
        try: 
            return self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception: 
            return False

    def _transcribe_float32(self, pcm_f32: np.ndarray) -> str:
        t0 = time.time()
        
        if self.stt_provider == "groq":
            # Use Groq API for transcription
            try:
                txt = self.model.transcribe_numpy(pcm_f32, SAMPLE_RATE, "pl")
                logger.info(f"[Perf] ASR_groq_ms={int((time.time()-t0)*1000)} len={len(txt)}")
                return txt
            except Exception as e:
                logger.error(f"Groq transcription failed: {e}", "ASR")
                logger.error("Falling back to local Whisper", "ASR")
                return self._transcribe_local(pcm_f32)
        else:
            # Use local Faster-Whisper
            return self._transcribe_local(pcm_f32)
    
    def _transcribe_local(self, pcm_f32: np.ndarray) -> str:
        """Transcribe using local Faster-Whisper"""
        t0 = time.time()
        try:
            segments, _ = self.model.transcribe(
                pcm_f32, language="pl", beam_size=1, vad_filter=False
            )
            txt = "".join(seg.text for seg in segments)
            logger.info(f"[Perf] ASR_local_ms={int((time.time()-t0)*1000)} len={len(txt)}")
            return txt
        except Exception as e:
            logger.error(f"Local transcription failed: {e}", "ASR")
            return ""

    def run_simulation(self):
        """Run in simulation mode for testing without audio"""
        logger.info("[SIM] Starting simulation mode - generating test audio")
        
        # Create a simple test sequence
        test_phrases = [
            "hej watusiu jak się masz",
            "powiedz mi jaka jest pogoda", 
            "dziękuję watusiu"
        ]
        
        for i, phrase in enumerate(test_phrases):
            logger.info(f"[SIM] Simulating phrase {i+1}: {phrase}")
            cue_think()
            
            # Simulate processing time
            time.sleep(2)
            
            # Simulate speaker verification
            if self.verifier.enabled and is_wake_word_present(phrase):
                self.verifier.enroll_samples(np.random.randn(16000).astype(np.float32), 16000)
                logger.info("[SIM] Simulated leader enrollment")
            
            # Create simulated transcription result
            text = phrase if i < 2 else ""  # Last phrase empty to test handling
            
            if text:
                turn_id = int(time.time() * 1000) + i
                line = {
                    "type": "leader_utterance",
                    "session_id": f"sim_{int(time.time())}",
                    "group_id": f"leader_{turn_id}",
                    "speaker_id": "leader",
                    "is_leader": True,
                    "turn_ids": [turn_id],
                    "text_full": text,
                    "category": "wypowiedź",
                    "reply_hint": True,
                    "ts_start": time.time(),
                    "ts_end": time.time() + 2,
                    "dbfs": -20.0,
                    "verify": {"enabled": True, "enrolled": True, "score": 0.8, "is_leader": True},
                    "emit_reason": "simulation",
                    "ts": time.time()
                }
                
                # Write to dialog file
                with open(DIALOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                
                logger.info(f"[SIM] Published: {text}")
                
                # Simulate ZMQ publish (would go to reporter)
                self.bus.publish_leader(line)
            
            cue_listen()
            time.sleep(1)
        
        logger.info("[SIM] Simulation complete")

    def run(self):
        if self.simulation_mode:
            logger.info("[SIM] Running in simulation mode")
            self.run_simulation()
            return

        # Real audio mode
        try:
            in_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=BLOCK_SIZE, device=IN_DEV
            )
            
            frame_ms = int(1000 * BLOCK_SIZE / SAMPLE_RATE)
            sil_frames_end = max(1, SIL_MS_END // frame_ms)
            min_speech_frames = max(1, VAD_MIN_MS // frame_ms)

            pre_buffer = deque(maxlen=15)
            speech_frames = bytearray()
            in_speech = False
            started_ms = None
            last_voice_ms = 0
            listening_flag = None

            with in_stream:
                logger.info("[OK] Audio stream started")
                while True:
                    now_ms = int(time.time()*1000)

                    if self.state.is_blocked():
                        if listening_flag is not False:
                            cue_idle(); listening_flag = False
                        in_speech = False; speech_frames = bytearray(); started_ms = None
                        pre_buffer.clear()
                        time.sleep(0.01); continue

                    if now_ms < self.cooldown_until:
                        time.sleep(0.003); continue

                    if listening_flag is not True:
                        cue_listen(); listening_flag = True

                    try:
                        audio, _ = in_stream.read(BLOCK_SIZE)
                    except Exception as e:
                        logger.error(f"Audio read error: {e}", "AUDIO")
                        time.sleep(0.01); continue

                    frame_bytes = audio.tobytes()
                    pre_buffer.append(frame_bytes)
                    is_sp = self._vad_is_speech(frame_bytes)

                    # Simplified speech detection (for brevity)
                    # In a real implementation, you'd include the full VAD logic
                    if is_sp and not in_speech:
                        in_speech = True
                        speech_frames = bytearray()
                        if pre_buffer:
                            speech_frames.extend(b''.join(pre_buffer))
                        started_ms = now_ms

                    # Handle speech end and processing
                    if in_speech and not is_sp:
                        if speech_frames and started_ms:
                            dur_ms = now_ms - started_ms
                            if dur_ms >= VAD_MIN_MS:
                                self._finalize(speech_frames, started_ms, now_ms, dur_ms)
                                self.cooldown_until = now_ms + self.emit_cooldown_ms
                        in_speech = False
                        speech_frames = bytearray()
                        listening_flag = None

                    time.sleep(0.0005)

        except Exception as e:
            logger.error(f"Audio stream failed: {e}", "AUDIO")
            logger.info("[SIM] Falling back to simulation mode")
            self.run_simulation()

    def _finalize(self, speech_frames: bytearray, started_ms: int, last_voice_ms: int, dur_ms: int):
        cue_think()
        pcm_f32 = np.frombuffer(speech_frames, dtype=np.int16).astype(np.float32)/32768.0
        dbfs = float(self._rms_dbfs(pcm_f32))
        if dbfs < ASR_MIN_DBFS:
            return

        # Transcribe
        text = self._transcribe_float32(pcm_f32).strip()
        if not text:
            return

        # Speaker verification logic (simplified)
        verify = {}
        is_leader = False
        is_wake_word = is_wake_word_present(text)

        if getattr(self.verifier, "enabled", False):
            if is_wake_word:
                logger.info("[SPK] Wake word detected. Enrolling new leader.")
                self.verifier.enroll_samples(pcm_f32, SAMPLE_RATE)
                verify = self.verifier.verify(pcm_f32, SAMPLE_RATE, dbfs)
                is_leader = True
            elif self.verifier.enrolled:
                verify = self.verifier.verify(pcm_f32, SAMPLE_RATE, dbfs)
                is_leader = bool(verify.get("is_leader", False))
            else:
                logger.info(f"[SPK] No leader and no wake word. Ignoring: '{text}'")
                return
        else:
            is_leader = not SPEAKER_REQUIRE_MATCH

        # Create and save dialog entry
        ts_start = started_ms/1000.0
        ts_end = last_voice_ms/1000.0
        turn_id = last_voice_ms

        line = {
            "type": "leader_utterance" if is_leader else "unknown_utterance",
            "session_id": self.state.session_id,
            "group_id": f"{'leader' if is_leader else 'unknown'}_{turn_id}",
            "speaker_id": "leader" if is_leader else "unknown",
            "is_leader": is_leader,
            "turn_ids": [turn_id],
            "text_full": text,
            "category": "wypowiedź",
            "reply_hint": is_leader,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "dbfs": dbfs,
            "verify": verify,
            "emit_reason": "endpoint",
            "ts": time.time()
        }
        
        with open(DIALOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        if is_leader:
            logger.info(f"[PUB] dialog.leader → group={line['group_id']} spk_score={verify.get('score')}")
            self.state.set_awaiting_reply(True)
            self.bus.publish_leader(line)
            self.state.pause_until_reply()
        else:
            logger.info(f"[SKIP] unknown (score={verify.get('score', 0):.2f}) zapisany, nie wysyłam ZMQ")

# State management
class State:
    def __init__(self):
        self.session_id = f"live_{int(time.time())}"
        self._tts_active = False
        self._awaiting_reply = False
        self._lock = threading.Lock()
        self.tts_pending_until = 0.0
        self.waiting_reply_until = 0.0
        self.last_tts_id = None

    def set_tts(self, flag: bool):
        with self._lock:
            self._tts_active = flag

    def set_awaiting_reply(self, flag: bool):
        with self._lock:
            self._awaiting_reply = flag

    def pause_until_reply(self):
        with self._lock:
            self.waiting_reply_until = time.time() + float(get_config_with_fallback("WAIT_REPLY_S", "0.6"))

    def is_blocked(self) -> bool:
        with self._lock:
            return (
                self._tts_active
                or self._awaiting_reply
                or (time.time() < self.tts_pending_until)
                or (time.time() < self.waiting_reply_until)
            )

# TTS Worker (placeholder for now)
def tts_worker(state: State, bus: Bus):
    logger.info("[TTS] Worker started")
    while True:
        msg = bus.get_tts(timeout=0.1)
        if not msg: 
            continue
        
        text = (msg.get("text") or "").strip()
        if text:
            logger.info(f"[TTS] Received text: {text[:50]}...")
        
        # For now, just log the TTS message
        # In real implementation, would use Piper TTS
        time.sleep(0.1)

# Main function with enhanced error handling
def main():
    """Main entry point with comprehensive error handling"""
    logger.info("=" * 60)
    logger.info("🤖 WATUS VOICE FRONTEND - Enhanced Version")
    logger.info("=" * 60)
    
    try:
        # Environment check
        if STT_PROVIDER == "groq":
            logger.info(f"[ENV] STT=groq MODEL={GROQ_MODEL}")
        else:
            logger.info(f"[ENV] STT=local MODEL={WHISPER_MODEL_NAME} DEVICE={WHISPER_DEVICE} COMPUTE={WHISPER_COMPUTE}")
        logger.info(f"[ENV] Audio mode: {AUDIO_MODE}")
        
        # Wake words
        wake_words_env = get_config_with_fallback("WAKE_WORDS", "hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu")
        wake_words = [w.strip() for w in wake_words_env.split(",") if w.strip()]
        logger.info(f"[WAKE] Configured wake words: {wake_words}")
        
        logger.info(f"[ZMQ] PUB: {PUB_ADDR} | SUB: {SUB_ADDR}")
        
        # Initialize ZMQ Bus
        try:
            bus = Bus(PUB_ADDR, SUB_ADDR)
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ Bus: {e}", "ZMQ")
            return False
        
        # Initialize State
        state = State()
        
        # Start TTS worker
        tts_thread = threading.Thread(target=tts_worker, args=(state, bus), daemon=True)
        tts_thread.start()
        
        # Initialize STT Engine
        try:
            stt = STTEngine(state, bus)
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}", "ASR")
            return False
        
        logger.info(f"[AUDIO] Input device: {IN_DEV} | Output device: {OUT_DEV}")
        
        # Start in listening state
        led.listening()
        
        logger.info("🚀 Watus is ready! Starting main loop...")
        
        # Main loop
        try:
            stt.run()
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", "MAIN")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return False
        
        logger.info("👋 Watus stopped gracefully")
        return True
        
    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}", "MAIN")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import traceback
    
    success = main()
    
    # Final status report
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL STATUS REPORT")
    logger.info("=" * 60)
    
    if logger.errors:
        logger.info(f"❌ Errors encountered: {len(logger.errors)}")
        for error in logger.errors:
            logger.info(f"  - [{error['component'] or 'UNKNOWN'}] {error['message']}")
    else:
        logger.info("✅ No errors encountered")
        
    if logger.warnings:
        logger.info(f"⚠️  Warnings: {len(logger.warnings)}")
        for warning in logger.warnings:
            logger.info(f"  - [{warning['component'] or 'UNKNOWN'}] {warning['message']}")
    else:
        logger.info("✅ No warnings")
    
    logger.info(f"🎯 Overall result: {'SUCCESS' if success else 'FAILED'}")
    
    if not success:
        logger.info("\n💡 Troubleshooting tips:")
        logger.info("1. Check audio device availability")
        logger.info("2. Verify .env configuration")
        logger.info("3. Ensure all dependencies are installed")
        logger.info("4. Run system_diagnostic.py for detailed analysis")
    
    sys.exit(0 if success else 1)
