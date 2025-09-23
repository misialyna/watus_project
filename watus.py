import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, json, time, queue, threading, subprocess, tempfile
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
import soundfile as sf
import zmq
import webrtcvad
from faster_whisper import WhisperModel

load_dotenv()

# ===== ZMQ =====
PUB_ADDR = os.environ.get("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")  # Watus:PUB.bind (dialog.leader/unknown_utterance)
SUB_ADDR = os.environ.get("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")  # Watus:SUB.connect (tts.speak)

# ===== Whisper / Piper =====
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE_TYPE", os.environ.get("WHISPER_COMPUTE", "auto"))

PIPER_BIN    = os.environ.get("PIPER_BIN")
PIPER_MODEL  = os.environ.get("PIPER_MODEL")
PIPER_CONFIG = os.environ.get("PIPER_CONFIG")

# ===== Audio/VAD =====
SAMPLE_RATE  = int(os.environ.get("WATUS_SR", "16000"))
BLOCK_SIZE   = int(os.environ.get("WATUS_BLOCKSIZE", str(int(round(SAMPLE_RATE*0.02)))))  # 20 ms default (troszkę szybciej)
VAD_MODE     = int(os.environ.get("WATUS_VAD_MODE", "1"))
VAD_MIN_MS   = int(os.environ.get("WATUS_VAD_MIN_MS", "280"))    # minimalnie krócej
SIL_MS_END   = int(os.environ.get("WATUS_SIL_MS_END", "650"))    # odcięcie ciszy trochę szybciej

IN_DEV_ENV  = os.environ.get("WATUS_INPUT_DEVICE")
OUT_DEV_ENV = os.environ.get("WATUS_OUTPUT_DEVICE")

DIALOG_PATH = os.environ.get("DIALOG_PATH", "dialog.jsonl")

# ===== Weryfikacja mówcy =====
SPEAKER_VERIFY            = int(os.environ.get("SPEAKER_VERIFY", "1"))
SPEAKER_ENROLL_MODE       = os.environ.get("SPEAKER_ENROLL_MODE", "first_good").strip().lower()
SPEAKER_THRESHOLD         = float(os.environ.get("SPEAKER_THRESHOLD", "0.64"))
SPEAKER_STICKY_THRESHOLD  = float(os.environ.get("SPEAKER_STICKY_THRESHOLD", str(SPEAKER_THRESHOLD)))
SPEAKER_GRACE             = float(os.environ.get("SPEAKER_GRACE", "0.08"))
SPEAKER_STICKY_SEC        = float(os.environ.get("SPEAKER_STICKY_SEC", os.environ.get("SPEAKER_STICKY_S", "300")))
SPEAKER_MIN_ENROLL_SCORE  = float(os.environ.get("SPEAKER_MIN_ENROLL_SCORE", "0.55"))
SPEAKER_MIN_DBFS          = float(os.environ.get("SPEAKER_MIN_DBFS", "-40"))
SPEAKER_MAX_DBFS          = float(os.environ.get("SPEAKER_MAX_DBFS", "-5"))
SPEAKER_BACK_THRESHOLD    = float(os.environ.get("SPEAKER_BACK_THRESHOLD", "0.58"))
SPEAKER_REQUIRE_MATCH     = int(os.environ.get("SPEAKER_REQUIRE_MATCH", "1"))

# ===== Zachowanie =====
WAIT_REPLY_S = float(os.environ.get("WAIT_REPLY_S", "2.0"))  # ile max czekamy na TTS zanim wrócimy do słuchania

def log(msg): print(msg, flush=True)

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
        self.pub.send_multipart([b"dialog.leader", json.dumps(payload, ensure_ascii=False).encode("utf-8")])

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
        self._lock = threading.Lock()
        self.tts_pending_until = 0.0
        self.waiting_reply_until = 0.0
        self.last_tts_id = None

    def set_tts(self, flag: bool):
        with self._lock:
            self._tts_active = flag

    def pause_until_reply(self):
        with self._lock:
            self.waiting_reply_until = time.time() + WAIT_REPLY_S

    def is_blocked(self) -> bool:
        with self._lock:
            return (
                self._tts_active
                or (time.time() < self.tts_pending_until)
                or (time.time() < self.waiting_reply_until)
            )

def cue_listen():  log("[Watus][STATE] LISTENING")
def cue_think():   log("[Watus][STATE] THINKING")
def cue_speak():   log("[Watus][STATE] SPEAKING")
def cue_idle():    log("[Watus][STATE] IDLE")

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
        import torch
        from speechbrain.inference import EncoderClassifier
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
            from speechbrain.inference import EncoderClassifier
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
            if age <= self.sticky_sec and sim >= (self.sticky_thr - SPEAKER_GRACE): is_leader=True
            elif sim >= SPEAKER_THRESHOLD: is_leader=True
            elif sim >= SPEAKER_BACK_THRESHOLD and age <= self.sticky_sec: is_leader=True
            return {"enabled": True, "enrolled": True, "score": sim, "is_leader": bool(is_leader), "sticky_age_s": age}

    return _SbVerifier()

# ===== Piper =====
def piper_say(text: str, out_dev=OUT_DEV):
    if not PIPER_BIN or not os.path.isfile(PIPER_BIN):
        log(f"[Watus][TTS] Brak/niepoprawny PIPER_BIN: {PIPER_BIN}"); return
    if not PIPER_MODEL or not os.path.isfile(PIPER_MODEL):
        log(f"[Watus][TTS] Brak/niepoprawny PIPER_MODEL: {PIPER_MODEL}"); return

    cfg = ["--config", PIPER_CONFIG] if PIPER_CONFIG and os.path.isfile(PIPER_CONFIG) else []
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = [PIPER_BIN, "--model", PIPER_MODEL, *cfg, "--output_file", wav_path]
        subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        data, sr = sf.read(wav_path, dtype="float32")
        sd.play(data, sr, device=out_dev, blocking=True)
    except subprocess.CalledProcessError as e:
        log(f"[Watus][TTS] Piper błąd: {e.stderr.decode('utf-8', 'ignore')}")
    except Exception as e:
        log(f"[Watus][TTS] Odtwarzanie nieudane: {e}")
    finally:
        try: os.unlink(wav_path)
        except Exception: pass

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
        # Szybka konfiguracja Whisper – beam_size=1 (najszybciej), bez temperature sampling
        self.model = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        self.verifier = _make_verifier()
        self.last_turn_id = None
        self.emit_cooldown_ms = 350   # krócej
        self.cooldown_until = 0

    @staticmethod
    def _rms_dbfs(x: np.ndarray, eps=1e-9):
        rms = np.sqrt(np.mean(np.square(x) + eps))
        return 20*np.log10(max(rms, eps))

    def _vad_is_speech(self, frame_bytes: bytes) -> bool:
        try: return self.vad.is_speech(frame_bytes, SAMPLE_RATE)
        except Exception: return False

    def _transcribe_float32(self, pcm_f32: np.ndarray) -> str:
        segments, _ = self.model.transcribe(pcm_f32, language="pl", beam_size=1, vad_filter=False)
        return "".join(seg.text for seg in segments)

    def run(self):
        in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=BLOCK_SIZE, device=IN_DEV
        )

        speech_frames = bytearray()
        in_speech = False
        started_ms = None
        last_voice_ms = 0
        listening_flag = None

        with in_stream:
            while True:
                now_ms = int(time.time()*1000)

                if self.state.is_blocked():
                    if listening_flag is not False:
                        cue_idle(); listening_flag = False
                    in_speech = False; speech_frames = bytearray(); started_ms = None
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
                is_sp = self._vad_is_speech(frame_bytes)

                if is_sp and not in_speech:
                    in_speech = True
                    speech_frames = bytearray()
                    started_ms = now_ms
                    last_voice_ms = now_ms

                if in_speech and is_sp:
                    speech_frames.extend(frame_bytes)
                    last_voice_ms = now_ms

                if in_speech and not is_sp:
                    if now_ms - last_voice_ms >= SIL_MS_END:
                        dur_ms = last_voice_ms - (started_ms or last_voice_ms)
                        in_speech = False

                        if dur_ms >= VAD_MIN_MS and len(speech_frames) > 0:
                            cue_think()
                            pcm_f32 = np.frombuffer(speech_frames, dtype=np.int16).astype(np.float32)/32768.0
                            dbfs = float(self._rms_dbfs(pcm_f32))

                            # enroll (first_good)
                            if (self.verifier.enabled and
                                self.verifier.__class__ != _NoopVerifier and
                                getattr(self.verifier, "enrolled", False) is False and
                                SPEAKER_ENROLL_MODE == "first_good"):
                                try: self.verifier.enroll_samples(pcm_f32, SAMPLE_RATE)
                                except Exception: pass

                            verify = self.verifier.verify(pcm_f32, SAMPLE_RATE, dbfs) if self.verifier else {"enabled": False}
                            is_leader = bool(verify.get("is_leader", False))

                            # transkrypcja
                            text = self._transcribe_float32(pcm_f32).strip()
                            if text:
                                ts_start = (started_ms or now_ms)/1000.0
                                ts_end   = last_voice_ms/1000.0
                                turn_id  = int(last_voice_ms)

                                line = {
                                    "type":"leader_utterance" if is_leader else "unknown_utterance",
                                    "session_id": self.state.session_id,
                                    "group_id": f"{'leader' if is_leader else 'unknown'}_{turn_id}",
                                    "speaker_id": "leader" if is_leader else "unknown",
                                    "is_leader": is_leader,
                                    "turn_ids":[turn_id],
                                    "text_full": text,
                                    "category":"wypowiedź",
                                    "reply_hint": is_leader,
                                    "ts_start": ts_start,
                                    "ts_end": ts_end,
                                    "dbfs": dbfs,
                                    "verify": verify,
                                    "emit_reason":"endpoint",
                                    "ts": time.time()
                                }
                                append_dialog_line(line, DIALOG_PATH)

                                if is_leader:
                                    log(f"[Watus][PUB] dialog.leader → group={line['group_id']}")
                                    self.bus.publish_leader(line)
                                    # chwilowo wstrzymaj nasłuch – czekamy na TTS
                                    self.state.pause_until_reply()
                                    self.state.tts_pending_until = time.time() + 0.8
                                else:
                                    log(f"[Watus][SKIP] unknown zapisany, nie wysyłam ZMQ")

                                self.cooldown_until = now_ms + self.emit_cooldown_ms

                            listening_flag = None
                            speech_frames = bytearray()
                            started_ms = None
                            last_voice_ms = now_ms

                time.sleep(0.001)

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

        # pokaż odpowiedź LLM w terminalu Watusa (krótka info – bez pełnego cytatu w logach STT)
        if text:
            log(f"[Watus][LLM] answer len={len(text)} (reply_to={reply_to})")

        state.set_tts(True); cue_speak()
        try:
            piper_say(text, out_dev=OUT_DEV)
        finally:
            state.set_tts(False); cue_listen()

# ===== Main =====
if __name__ == "__main__":
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
    try:
        stt.run()
    except KeyboardInterrupt:
        log("[Watus] stop"); sys.exit(0)
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"[Watus] fatal: {e}"); sys.exit(1)
