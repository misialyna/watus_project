#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CT2_SKIP_CONVERTERS", "1")  # <— kluczowe: pomija import transformers w ctranslate2

import sys, json, time, queue, threading, subprocess, tempfile, atexit, re
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import soundfile as sf
import zmq
import webrtcvad

# === ASR: Faster-Whisper (ONLY) ===
from faster_whisper import WhisperModel

# === Kontroler diody LED ===
from led_controller import LEDController

# .env z katalogu pliku (i override, gdy IDE ma własne env)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# ===== ZMQ =====
PUB_ADDR = os.environ.get("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")  # Watus:PUB.bind (dialog.leader/unknown_utterance)
SUB_ADDR = os.environ.get("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")  # Watus:SUB.connect (tts.speak)

# ===== Whisper / Piper =====
def _normalize_fw_model(name: str) -> str:
    name = (name or "").strip()
    short = {"tiny","base","small","medium","large","large-v1","large-v2","large-v3"}
    if "/" not in name and name.lower() in short:
        return f"guillaumekln/faster-whisper-{name.lower()}"
    return name

WHISPER_MODEL_NAME = _normalize_fw_model(os.environ.get("WHISPER_MODEL", "small"))
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE_TYPE", os.environ.get("WHISPER_COMPUTE", "int8"))
CPU_THREADS        = int(os.environ.get("WATUS_CPU_THREADS", str(os.cpu_count() or 4)))
WHISPER_NUM_WORKERS= int(os.environ.get("WHISPER_NUM_WORKERS", "1"))

PIPER_BIN    = os.environ.get("PIPER_BIN")
PIPER_MODEL  = os.environ.get("PIPER_MODEL")
PIPER_CONFIG = os.environ.get("PIPER_CONFIG")
PIPER_SR     = os.environ.get("PIPER_SAMPLE_RATE")

# ===== Audio/VAD =====
SAMPLE_RATE  = int(os.environ.get("WATUS_SR", "16000"))
BLOCK_SIZE   = int(os.environ.get("WATUS_BLOCKSIZE", str(int(round(SAMPLE_RATE*0.02)))))  # ~20 ms
VAD_MODE     = int(os.environ.get("WATUS_VAD_MODE", "1"))
VAD_MIN_MS   = int(os.environ.get("WATUS_VAD_MIN_MS", "150")) # ZMNIEJSZONO z 280ms dla krótkich słów
SIL_MS_END   = int(os.environ.get("WATUS_SIL_MS_END", "450")) # ZMNIEJSZONO z 650ms dla szybszej reakcji
ASR_MIN_DBFS = float(os.environ.get("ASR_MIN_DBFS", "-34"))

# endpoint anti-chop
PREBUFFER_FRAMES = int(os.environ.get("WATUS_PREBUFFER_FRAMES", "15")) # 15 ramek * 20ms = 300ms
START_MIN_FRAMES = int(os.environ.get("WATUS_START_MIN_FRAMES", "4")) # 4 ramki * 20ms = 80ms
START_MIN_DBFS   = float(os.environ.get("WATUS_START_MIN_DBFS", str(ASR_MIN_DBFS + 4.0)))
MIN_MS_BEFORE_ENDPOINT = int(os.environ.get("WATUS_MIN_MS_BEFORE_ENDPOINT", "500"))
END_AT_DBFS_DROP = float(os.environ.get("END_AT_DBFS_DROP", "0"))
EMIT_COOLDOWN_MS = int(os.environ.get("EMIT_COOLDOWN_MS", "300"))
MAX_UTT_MS       = int(os.environ.get("MAX_UTT_MS", "6500"))
GAP_TOL_MS       = int(os.environ.get("WATUS_GAP_TOL_MS", "450"))  # cisza, którą jeszcze tolerujemy w środku wypowiedzi

IN_DEV_ENV  = os.environ.get("WATUS_INPUT_DEVICE")
OUT_DEV_ENV = os.environ.get("WATUS_OUTPUT_DEVICE")

DIALOG_PATH = os.environ.get("DIALOG_PATH", "dialog.jsonl")

# ===== Weryfikacja mówcy =====
SPEAKER_VERIFY            = int(os.environ.get("SPEAKER_VERIFY", "1"))
WAKE_WORDS                = [w.strip() for w in os.environ.get("WAKE_WORDS", "hej watusiu,hej watuszu,hej watusił,kej watusił,hej watośiu").split(",") if w.strip()]
SPEAKER_THRESHOLD         = float(os.environ.get("SPEAKER_THRESHOLD", "0.64"))
SPEAKER_STICKY_THRESHOLD  = float(os.environ.get("SPEAKER_STICKY_THRESHOLD", str(SPEAKER_THRESHOLD)))
SPEAKER_GRACE             = float(os.environ.get("SPEAKER_GRACE", "0.12"))  # lekko w górę – emocje
SPEAKER_STICKY_SEC        = float(os.environ.get("SPEAKER_STICKY_SEC", os.environ.get("SPEAKER_STICKY_S", "3600")))
SPEAKER_MIN_ENROLL_SCORE  = float(os.environ.get("SPEAKER_MIN_ENROLL_SCORE", "0.55"))
SPEAKER_MIN_DBFS          = float(os.environ.get("SPEAKER_MIN_DBFS", "-40"))
SPEAKER_MAX_DBFS          = float(os.environ.get("SPEAKER_MAX_DBFS", "-5"))
SPEAKER_BACK_THRESHOLD    = float(os.environ.get("SPEAKER_BACK_THRESHOLD", "0.56"))
SPEAKER_REQUIRE_MATCH     = int(os.environ.get("SPEAKER_REQUIRE_MATCH", "1"))

# ===== Zachowanie =====
WAIT_REPLY_S = float(os.environ.get("WAIT_REPLY_S", "0.6"))  # max czekania na TTS zanim wrócimy do słuchania

def log(msg): print(msg, flush=True)

def is_wake_word_present(text: str) -> bool:
    """
    Sprawdza obecność słowa-klucza w bardziej elastyczny sposób.
    1. Normalizuje tekst wejściowy (małe litery, usuwa znaki interpunkcyjne).
    2. Sprawdza, czy którakolwiek z fraz kluczowych (również znormalizowana) znajduje się w tekście.
    """
    normalized_text = re.sub(r'[^\w\s]', '', text.lower())
    
    for wake_phrase in WAKE_WORDS:
        normalized_wake_phrase = re.sub(r'[^\w\s]', '', wake_phrase.lower())
        if normalized_wake_phrase in normalized_text:
            return True
    return False

# ===== LED Controller =====
led = LEDController()
atexit.register(led.cleanup)

def list_devices():
    try:
        devs = sd.query_devices()
        log("[Watus] Audio devices:")
        for i, d in enumerate(devs):
            log(f"  [{i}] {d['name']} in:{d.get('max_input_channels',0)} out:{d.get('max_output_channels',0)}")
    except Exception as e:
        log(f"[Watus] Nie można wypisać urządzeń: {e}")

def _match_device_by_name(fragment: str, want_output=False):
    frag = (fragment or "").lower().strip()
    for i, d in enumerate(sd.query_devices()):
        if frag and frag in d['name'].lower():
            if want_output and d.get("max_output_channels", 0) > 0: return i
            if not want_output and d.get("max_input_channels", 0) > 0: return i
    return None

def _coerce_dev(v, want_output=False):
    if not v: return None
    try: return int(v)
    except ValueError: return _match_device_by_name(v, want_output=want_output)

def _auto_pick_input():
    try:
        di = sd.default.device
        if isinstance(di, (list, tuple)) and di[0] is not None: return di[0]
    except Exception: pass
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0: return i
    return None

def _auto_pick_output():
    try:
        di = sd.default.device
        if isinstance(di, (list, tuple)) and di[1] is not None: return di[1]
    except Exception: pass
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0: return i
    return None

IN_DEV  = _coerce_dev(IN_DEV_ENV, want_output=False) or _auto_pick_input()
OUT_DEV = _coerce_dev(OUT_DEV_ENV, want_output=True)  or _auto_pick_output()

# ===== ZMQ Bus =====
class Bus:
    def __init__(self, pub_addr: str, sub_addr: str):
        ctx = zmq.Context.instance()
        self.pub = ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 100)
        self.pub.bind(pub_addr)

        self.sub = ctx.socket(zmq.SUB)
        self.sub.connect(sub_addr)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "tts.speak")

        self._sub_queue = queue.Queue()
        threading.Thread(target=self._sub_loop, daemon=True).start()

    def publish_leader(self, payload: dict):
        t0 = time.time()
        self.pub.send_multipart([b"dialog.leader", json.dumps(payload, ensure_ascii=False).encode("utf-8")])
        log(f"[Perf] BUS_ms={int((time.time()-t0)*1000)}")

    def _sub_loop(self):
        while True:
            try:
                topic, payload = self.sub.recv_multipart()
                if topic != b"tts.speak": continue
                data = json.loads(payload.decode("utf-8", "ignore"))
                self._sub_queue.put(data)
            except Exception:
                time.sleep(0.01)

    def get_tts(self, timeout=0.1):
        try: return self._sub_queue.get(timeout=timeout)
        except queue.Empty: return None

# ===== Stan =====
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
            self.waiting_reply_until = time.time() + WAIT_REPLY_S

    def is_blocked(self) -> bool:
        with self._lock:
            return (
                self._tts_active
                or self._awaiting_reply
                or (time.time() < self.tts_pending_until)
                or (time.time() < self.waiting_reply_until)
            )

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

# ===== Weryfikator (ECAPA) =====
class _NoopVerifier:
    enabled=True
    def __init__(self): self._enrolled=None
    @property
    def enrolled(self): return False
    def enroll_wav(self, p): pass
    def enroll_samples(self, s, sr): pass
    def verify(self, s, sr, db): return {"enabled": False}

def _make_verifier():
    if not SPEAKER_VERIFY: return _NoopVerifier()
    try:
        import torch  # noqa
        from speechbrain.pretrained import EncoderClassifier  # noqa
    except Exception as e:
        log(f"[Watus][SPK] OFF (brak zależności): {e}")
        return _NoopVerifier()

    class _SbVerifier:
        enabled=True
        def __init__(self):
            import torch
            self.threshold   = SPEAKER_THRESHOLD
            self.sticky_thr  = SPEAKER_STICKY_THRESHOLD
            self.back_thr    = SPEAKER_BACK_THRESHOLD
            self.grace       = SPEAKER_GRACE
            self.sticky_sec  = SPEAKER_STICKY_SEC
            self._clf=None
            self._device="cuda" if torch.cuda.is_available() else "cpu"
            self._enrolled=None
            self._enroll_ts=0.0

        @property
        def enrolled(self): return self._enrolled is not None

        def _ensure(self):
            from speechbrain.pretrained import EncoderClassifier
            if self._clf is None:
                self._clf = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": self._device},
                    savedir="models/ecapa",
                )

        @staticmethod
        def _resample_16k(x: np.ndarray, sr: int) -> np.ndarray:
            if sr == 16000: return x.astype(np.float32)
            ratio = 16000.0/sr
            n_out = int(round(len(x)*ratio))
            idx = np.linspace(0, len(x)-1, num=n_out, dtype=np.float32)
            base = np.arange(len(x), dtype=np.float32)
            return np.interp(idx, base, x).astype(np.float32)

        def _embed(self, samples: np.ndarray, sr: int):
            import torch
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
                log(f"[Watus][SPK] Enrolled new leader voice.")
            except Exception as e:
                log(f"[Watus][SPK] enroll err: {e}")

        def verify(self, samples: np.ndarray, sr: int, dbfs: float) -> dict:
            if self._enrolled is None:
                return {"enabled": True, "enrolled": False}
            import torch, torch.nn.functional as F
            a = self._embed(samples, sr)
            sim = float(F.cosine_similarity(
                torch.tensor(a, dtype=torch.float32).flatten(),
                torch.tensor(self._enrolled, dtype=torch.float32).flatten(), dim=0, eps=1e-8
            ).detach().cpu().item())
            now = time.time()
            age = now - self._enroll_ts
            is_leader = False
            adj_thr = (self.sticky_thr - self.grace) if dbfs > -22.0 else self.sticky_thr  # emocje → głośniej → trochę łagodniej
            if age <= self.sticky_sec and sim >= adj_thr: is_leader=True
            elif sim >= self.threshold: is_leader=True
            elif sim >= self.back_thr and age <= self.sticky_sec: is_leader=True
            return {"enabled": True, "enrolled": True, "score": sim, "is_leader": bool(is_leader), "sticky_age_s": age}

    return _SbVerifier()

# ===== Piper CLI =====
def _env_with_libs_for_piper(piper_bin: str) -> dict:
    env = os.environ.copy()
    bin_dir = os.path.dirname(piper_bin) if piper_bin else ""
    phonemize_lib = os.path.join(bin_dir, "piper-phonemize", "lib")
    extra_paths = []
    if os.path.isdir(bin_dir): extra_paths.append(bin_dir)
    if os.path.isdir(phonemize_lib): extra_paths.append(phonemize_lib)
    if not extra_paths: return env

    if sys.platform == "darwin":
        key = "DYLD_LIBRARY_PATH"
    elif sys.platform.startswith("linux"):
        key = "LD_LIBRARY_PATH"
    else:
        key = "PATH"
    cur = env.get(key, "")
    sep = (":" if key != "PATH" else ";")
    env[key] = (sep.join([*extra_paths, cur]) if cur else sep.join(extra_paths))
    return env

def piper_say(text: str, out_dev=OUT_DEV):
    if not text or not text.strip(): return
    if not PIPER_BIN or not os.path.isfile(PIPER_BIN):
        log(f"[Watus][TTS] Uwaga: brak/niepoprawny PIPER_BIN: {PIPER_BIN}"); return
    if not PIPER_MODEL or not os.path.isfile(PIPER_MODEL):
        log(f"[Watus][TTS] Brak/niepoprawny PIPER_MODEL: {PIPER_MODEL}"); return

    try:
        if sys.platform == "darwin":
            bin_dir = os.path.dirname(PIPER_BIN)
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", bin_dir],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

    cfg = ["--config", PIPER_CONFIG] if PIPER_CONFIG and os.path.isfile(PIPER_CONFIG) else []
    env = _env_with_libs_for_piper(PIPER_BIN)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    t0 = time.time()
    try:
        cmd = [PIPER_BIN, "--model", PIPER_MODEL, *cfg, "--output_file", wav_path]
        if PIPER_SR: cmd += ["--sample_rate", str(PIPER_SR)]
        subprocess.run(cmd, input=(text or "").encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=env)
        data, sr = sf.read(wav_path, dtype="float32")
        sd.play(data, sr, device=out_dev, blocking=True)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        log(f"[Watus][TTS] Piper błąd (proc): {err}")
    except Exception as e:
        log(f"[Watus][TTS] Odtwarzanie nieudane: {e}")
    finally:
        try: os.unlink(wav_path)
        except Exception: pass
        log(f"[Perf] TTS_play_ms={int((time.time()-t0)*1000)}")

# ===== JSONL =====
def append_dialog_line(obj: dict, path=DIALOG_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ===== STT =====
class STTEngine:
    def __init__(self, state: 'State', bus: 'Bus'):
        self.state = state
        self.bus = bus
        self.vad = webrtcvad.Vad(VAD_MODE)
        log(f"[Watus] STT ready (device={IN_DEV} sr={SAMPLE_RATE} block={BLOCK_SIZE})")

        # Faster-Whisper ONLY
        log(f"[Watus] FasterWhisper init: model={WHISPER_MODEL_NAME} device={WHISPER_DEVICE} "
            f"compute={WHISPER_COMPUTE} cpu_threads={CPU_THREADS} workers={WHISPER_NUM_WORKERS}")
        t0 = time.time()
        self.model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
            cpu_threads=CPU_THREADS,
            num_workers=WHISPER_NUM_WORKERS
        )
        log(f"[Watus] STT FasterWhisper loaded ({int((time.time()-t0)*1000)} ms)")
        self.verifier = _make_verifier()
        self.emit_cooldown_ms = EMIT_COOLDOWN_MS
        self.cooldown_until = 0

    @staticmethod
    def _rms_dbfs(x: np.ndarray, eps=1e-9):
        rms = np.sqrt(np.mean(np.square(x) + eps))
        return 20*np.log10(max(rms, eps))

    def _vad_is_speech(self, frame_bytes: bytes) -> bool:
        try: return self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception: return False

    def _transcribe_float32(self, pcm_f32: np.ndarray) -> str:
        t0 = time.time()
        segments, _ = self.model.transcribe(
            pcm_f32, language="pl", beam_size=1, vad_filter=False
        )
        txt = "".join(seg.text for seg in segments)
        log(f"[Perf] ASR_ms={int((time.time()-t0)*1000)} len={len(txt)}")
        return txt

    def run(self):
        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=BLOCK_SIZE, device=IN_DEV
        )

        frame_ms = int(1000 * BLOCK_SIZE / SAMPLE_RATE)
        sil_frames_end     = max(1, SIL_MS_END // frame_ms)
        min_speech_frames  = max(1, VAD_MIN_MS  // frame_ms)

        pre_buffer = deque(maxlen=PREBUFFER_FRAMES)
        speech_frames = bytearray()
        in_speech = False
        started_ms = None
        last_voice_ms = 0
        listening_flag = None

        # start detection
        start_voice_run = 0

        # for anti-drop
        last_dbfs = None
        speech_frames_count = 0
        silence_run = 0
        gap_run = 0

        with in_stream:
            while True:
                now_ms = int(time.time()*1000)

                if self.state.is_blocked():
                    if listening_flag is not False:
                        cue_idle(); listening_flag = False
                    in_speech = False; speech_frames = bytearray(); started_ms = None
                    start_voice_run = 0; last_dbfs = None
                    speech_frames_count = 0; silence_run = 0; gap_run = 0
                    pre_buffer.clear()
                    time.sleep(0.01); continue

                if now_ms < self.cooldown_until:
                    time.sleep(0.003); continue

                if listening_flag is not True:
                    cue_listen(); listening_flag = True

                try:
                    audio, _ = in_stream.read(BLOCK_SIZE)
                except Exception as e:
                    log(f"[Watus][STT] read err: {e}")
                    time.sleep(0.01); continue

                frame_bytes = audio.tobytes()
                pre_buffer.append(frame_bytes)
                is_sp = self._vad_is_speech(frame_bytes)

                # --- twardy start: kilka ramek powyżej progu dBFS ---
                if not in_speech:
                    if is_sp:
                        cur = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        cur_db = float(self._rms_dbfs(cur))
                        if cur_db > START_MIN_DBFS:
                            start_voice_run += 1
                        else:
                            start_voice_run = 0
                        if start_voice_run >= START_MIN_FRAMES:
                            in_speech = True
                            speech_frames = bytearray()
                            if pre_buffer:
                                speech_frames.extend(b''.join(pre_buffer))
                            started_ms = now_ms - (len(pre_buffer) * frame_ms)
                            last_voice_ms = now_ms
                            speech_frames_count = 0
                            silence_run = 0
                            gap_run = 0
                            last_dbfs = None
                            start_voice_run = 0
                    else:
                        start_voice_run = 0
                    time.sleep(0.0005)
                    continue

                # --- w turze mowy ---
                if is_sp:
                    speech_frames.extend(frame_bytes)
                    last_voice_ms = now_ms
                    silence_run = 0
                    gap_run = 0
                    speech_frames_count += 1

                    cur = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    cur_db = float(self._rms_dbfs(cur))
                    if last_dbfs is None: last_dbfs = cur_db

                    if END_AT_DBFS_DROP > 0:
                        if speech_frames_count >= min_speech_frames and (now_ms - (started_ms or now_ms)) >= MIN_MS_BEFORE_ENDPOINT:
                            if (last_dbfs - cur_db) >= END_AT_DBFS_DROP:
                                in_speech = False
                                dur_ms = last_voice_ms - (started_ms or last_voice_ms)
                                self._finalize(speech_frames, started_ms, last_voice_ms, dur_ms)
                                listening_flag = None
                                speech_frames = bytearray(); started_ms = None
                                speech_frames_count = 0; silence_run = 0; last_dbfs = None; gap_run = 0
                                self.cooldown_until = now_ms + self.emit_cooldown_ms
                                continue
                    else:
                        last_dbfs = cur_db

                else:
                    # brak VAD -> liczymy ciszę i tolerowany GAP
                    silence_run += 1
                    gap_run += frame_ms
                    # toleruj krótką przerwę w środku wypowiedzi
                    if gap_run < GAP_TOL_MS:
                        speech_frames.extend(frame_bytes) # dodajemy ciszę do bufora na wszelki wypadek
                        continue

                    if silence_run >= sil_frames_end and (now_ms - (started_ms or now_ms)) >= MIN_MS_BEFORE_ENDPOINT:
                        in_speech = False
                        dur_ms = last_voice_ms - (started_ms or last_voice_ms)
                        if dur_ms >= VAD_MIN_MS and len(speech_frames) > 0:
                            self._finalize(speech_frames, started_ms, last_voice_ms, dur_ms)
                            self.cooldown_until = now_ms + self.emit_cooldown_ms
                        listening_flag = None
                        speech_frames = bytearray(); started_ms = None
                        speech_frames_count = 0; silence_run = 0; last_dbfs = None; gap_run = 0

                # twardy limit
                if in_speech and started_ms and (now_ms - started_ms) >= MAX_UTT_MS:
                    in_speech = False
                    dur_ms = last_voice_ms - (started_ms or last_voice_ms)
                    if dur_ms >= VAD_MIN_MS and len(speech_frames) > 0:
                        self._finalize(speech_frames, started_ms, last_voice_ms, dur_ms)
                        self.cooldown_until = now_ms + self.emit_cooldown_ms
                    listening_flag = None
                    speech_frames = bytearray(); started_ms = None
                    speech_frames_count = 0; silence_run = 0; last_dbfs = None; gap_run = 0

                time.sleep(0.0005)

    def _finalize(self, speech_frames: bytearray, started_ms: int, last_voice_ms: int, dur_ms: int):
        cue_think()
        pcm_f32 = np.frombuffer(speech_frames, dtype=np.int16).astype(np.float32)/32768.0
        dbfs = float(self._rms_dbfs(pcm_f32))
        if dbfs < ASR_MIN_DBFS:
            return

        # 1. Transkrypcja
        text = self._transcribe_float32(pcm_f32).strip()
        if not text:
            return

        # 2. Logika Lidera oparta na słowie-klucz
        verify = {}
        is_leader = False
        is_wake_word = is_wake_word_present(text)

        if getattr(self.verifier, "enabled", False):
            if is_wake_word:
                log("[Watus][SPK] Wykryto słowo-klucz. Rejestrowanie nowego lidera.")
                self.verifier.enroll_samples(pcm_f32, SAMPLE_RATE)
                verify = self.verifier.verify(pcm_f32, SAMPLE_RATE, dbfs)
                is_leader = True
            elif self.verifier.enrolled:
                verify = self.verifier.verify(pcm_f32, SAMPLE_RATE, dbfs)
                is_leader = bool(verify.get("is_leader", False))
            else:
                log(f"[Watus][SPK] Brak lidera i słowa-klucz. Ignoruję: '{text}'")
                return
        else:
            # Jeśli weryfikacja jest wyłączona, każda wypowiedź jest od "lidera"
            is_leader = not SPEAKER_REQUIRE_MATCH

        # 3. Przygotowanie i wysłanie danych
        ts_start = (started_ms or last_voice_ms)/1000.0
        ts_end   = last_voice_ms/1000.0
        turn_id  = int(last_voice_ms)

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
        append_dialog_line(line, DIALOG_PATH)

        if is_leader:
            log(f"[Watus][PUB] dialog.leader → group={line['group_id']} spk_score={verify.get('score')}")
            self.state.set_awaiting_reply(True)
            self.bus.publish_leader(line)
            self.state.pause_until_reply()
            self.state.tts_pending_until = time.time() + 0.6
        else:
            log(f"[Watus][SKIP] unknown (score={verify.get('score', 0):.2f}) zapisany, nie wysyłam ZMQ")


# ===== TTS worker =====
def tts_worker(state: State, bus: Bus):
    log("[Watus] Piper ready.")
    while True:
        msg = bus.get_tts(timeout=0.1)
        if not msg: continue
        text = (msg.get("text") or "").strip()
        reply_to = msg.get("reply_to") or ""
        if state.last_tts_id == reply_to and reply_to:
            log(f"[Watus][SUB] tts.speak DUP reply_to={reply_to} – pomijam")
            continue
        state.last_tts_id = reply_to

        if text:
            log(f"[Watus][LLM] answer len={len(text)} (reply_to={reply_to})")

        # OD TERAZ prawdziwy TTS – blokujemy słuchanie
        state.set_awaiting_reply(False)
        state.set_tts(True); cue_speak()
        try:
            piper_say(text, out_dev=OUT_DEV)
        finally:
            state.set_tts(False); cue_listen()

# ===== Main =====
if __name__ == "__main__":
    log(f"[Env] ASR=Faster WHISPER_MODEL={WHISPER_MODEL_NAME} WHISPER_DEVICE={WHISPER_DEVICE} "
        f"WHISPER_COMPUTE={WHISPER_COMPUTE} WATUS_BLOCKSIZE={BLOCK_SIZE}")
    log(f"[Watus] Wake words: {WAKE_WORDS}")
    log(f"[Watus] PUB dialog.leader @ {PUB_ADDR} | SUB tts.speak @ {SUB_ADDR}")
    list_devices()
    bus = Bus(PUB_ADDR, SUB_ADDR)
    state = State()
    threading.Thread(target=tts_worker, args=(state, bus), daemon=True).start()

    try:
        stt = STTEngine(state, bus)
    except Exception as e:
        log(f"[Watus] STT init error: {e}"); sys.exit(1)

    log(f"[Watus] IO: input={IN_DEV!r} | output={OUT_DEV!r}")
    led.listening() # Start with listening state
    try:
        stt.run()
    except KeyboardInterrupt:
        log("[Watus] stop"); sys.exit(0)
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"[Watus] fatal: {e}"); sys.exit(1)