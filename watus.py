#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Watuś — STT + diarization (offline), wake-word, TTS, heurystyczne emocje.
Tryby: batch (pliki), live (mikrofon). Logi JSON: logs.json (nadpisywane).
"""

import os, sys, json, math, time, yaml, argparse, logging, subprocess, warnings, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

# środowisko / stabilność CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# zależności
import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import sounddevice as sd
import pyttsx3

from concurrent.futures import ProcessPoolExecutor, as_completed
from faster_whisper import WhisperModel

# sklearn (klasteryzacja)
from sklearn.cluster import AgglomerativeClustering

# Resemblyzer (opcjonalnie — 100% offline jeśli weights są lokalnie)
RESEMBLYZER_AVAILABLE = True
try:
    from resemblyzer import VoiceEncoder
except Exception:
    RESEMBLYZER_AVAILABLE = False

# =========================
# KONFIG
# =========================

@dataclass
class Config:
    # ASR
    model: str
    device: str
    compute_type: str
    language: str
    hotwords: List[str]
    vad_min_silence_ms: int

    # I/O
    in_dir: str
    tmp_dir: str
    export_formats: List[str]
    batch_workers: int

    # FFmpeg
    ffmpeg_bin_env: str
    ffmpeg_af: str
    denoise: bool

    # Diarization (Resemblyzer -> fallback MFCC)
    diarize: bool
    resemblyzer_weights_path: str
    cluster_distance: float
    min_seg_duration: float

    # LIVE / wake-word
    wake_phrases: List[str]
    live_sample_rate: int
    live_frame_ms: int
    live_max_pre_roll_sec: float
    live_silence_after_wake_ms: int
    live_min_speech_ms: int
    live_input_device: Optional[int]

    # progi energii (dBFS)
    wake_probe_min_dbfs: float
    end_energy_dbfs: float

    # Słownik poprawek
    proper_name_fixes: Dict[str, str]

    # „pamięć” (opcjonalna etykieta stała na bazie podobieństwa)
    speaker_match_threshold: float

    # logi
    log_level: str
    logs_path: str

def load_config(path: Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    # ENV override
    y["model"] = os.getenv("WATUS_MODEL", y["model"])
    y["device"] = os.getenv("WATUS_DEVICE", y["device"])
    y["compute_type"] = os.getenv("WATUS_COMPUTE_TYPE", y["compute_type"])

    # domyślne klucze, jeśli nie ma w YAML
    y.setdefault("live_input_device", None)
    y.setdefault("wake_probe_min_dbfs", -45.0)
    y.setdefault("end_energy_dbfs", -45.0)

    return Config(**y)

# =========================
# LOGGING
# =========================

def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ["torch", "ctranslate2", "hugginface_hub", "urllib3", "numba", "resemblyzer"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# FFmpeg utils
# =========================

def check_ffmpeg(ffmpeg_bin_env: str) -> str:
    ffmpeg_bin = os.getenv(ffmpeg_bin_env) or "ffmpeg"
    try:
        subprocess.run([ffmpeg_bin, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return ffmpeg_bin
    except Exception:
        logging.error("ffmpeg nie jest dostępny. Zainstaluj: brew install ffmpeg "
                      "albo ustaw %s=/pełna/ścieżka/do/ffmpeg", ffmpeg_bin_env)
        raise

def ffmpeg_denoise_to_wav(ffmpeg_bin: str, src_path: Path, tmp_dir: Path, af: str) -> Path:
    out_path = tmp_dir / f"{src_path.stem}_denoise.wav"
    cmd = [ffmpeg_bin, "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", "-vn", "-af", af, str(out_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logging.exception("ffmpeg nie powiódł się dla %s (kod %s).", src_path.name, e.returncode)
        raise
    return out_path

# =========================
# Audio pomocnicze
# =========================

def dbfs(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12)
    return 20.0 * math.log10(rms + 1e-12)

def estimate_ambient_dbfs(y: np.ndarray, frame_len: int = 1024, hop: int = 512) -> float:
    frames = [y[i:i+frame_len] for i in range(0, max(1, len(y)-frame_len), hop)]
    levels = [dbfs(fr) for fr in frames if len(fr) == frame_len]
    return float(np.percentile(levels, 20)) if levels else dbfs(y)

def pretty_timestamp(t: float) -> str:
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}"

SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma"}

# =========================
# TTS (polski męski)
# =========================

def tts_say(text: str):
    try:
        engine = pyttsx3.init()
        target = None
        for v in engine.getProperty("voices"):
            name = (v.name or "").lower()
            lang = ";".join(getattr(v, "languages", []) or []).lower()
            if ("pl" in lang or "pol" in name) and ("m" in (getattr(v, "gender", "m") or "m").lower() or "miko" in name):
                target = v.id; break
        if target:
            engine.setProperty("voice", target)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        logging.exception("TTS błąd")

# =========================
# Whisper
# =========================

def load_whisper(model: str, device: str, compute_type: str) -> WhisperModel:
    logging.info("Ładuję Whisper: %s (device=%s, compute=%s)", model, device, compute_type)
    return WhisperModel(model, device=device, compute_type=compute_type)

def apply_name_fixes(text: str, fixes: Dict[str, str]) -> str:
    out = text
    for wrong, correct in fixes.items():
        out = out.replace(wrong, correct)
        out = out.replace(wrong.capitalize(), correct)
    return out

# =========================
# Emocje (heurystyki)
# =========================

def estimate_pitch(y: np.ndarray, sr: int) -> Tuple[float, float]:
    try:
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            return 0.0, 0.0
        return float(np.median(f0)), float(np.std(f0))
    except Exception:
        return 0.0, 0.0

def emotion_from_features(rms_db: float, f0_med: float, f0_std: float, wps: float) -> str:
    if rms_db > -20 and (wps > 3.5 or f0_std > 30):
        return "angry"
    if rms_db > -25 and (f0_med > 180 or wps > 2.5):
        return "happy"
    if rms_db < -30 and f0_med < 130 and wps < 1.5:
        return "sad"
    return "neutral"

# =========================
# Diarization: Resemblyzer → clustering (fallback: MFCC)
# =========================

def segment_embeddings(y: np.ndarray, sr: int, segments: List[Dict], cfg: Config) -> Tuple[np.ndarray, List[int]]:
    sel_idx, clips = [], []
    for i, s in enumerate(segments):
        dur = max(0.0, float(s["end"]) - float(s["start"]))
        if dur < cfg.min_seg_duration:
            continue
        a = int(s["start"] * sr); b = int(s["end"] * sr)
        seg = y[max(0, a):min(len(y), b)]
        if seg.size == 0:
            continue
        clips.append(seg)
        sel_idx.append(i)

    if not clips:
        return np.zeros((0, 256), dtype=np.float32), sel_idx

    if RESEMBLYZER_AVAILABLE:
        try:
            enc = VoiceEncoder(weights_fpath=cfg.resemblyzer_weights_path) if cfg.resemblyzer_weights_path else VoiceEncoder()
            embs = [enc.embed_utterance(c.astype(np.float32)) for c in clips]
            return np.vstack(embs).astype(np.float32), sel_idx
        except Exception:
            logging.warning("Resemblyzer niedostępny (brak wag/cuda). Używam fallback MFCC.")

    embs = []
    for c in clips:
        mf = librosa.feature.mfcc(y=c, sr=sr, n_mfcc=20)
        d = librosa.feature.delta(mf)
        feat = np.concatenate([mf.mean(axis=1), d.mean(axis=1)], axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-9)
        embs.append(feat.astype(np.float32))
    return np.vstack(embs), sel_idx

def cluster_speakers(embs: np.ndarray, distance: float) -> np.ndarray:
    if embs.size == 0:
        return np.array([], dtype=int)
    if embs.shape[0] == 1:
        return np.array([0], dtype=int)
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance,
            affinity="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embs)
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embs)
    return labels

def assign_cluster_labels(segments: List[Dict], sel_idx: List[int], labels: List[int]) -> None:
    for idx, lab in zip(sel_idx, labels):
        segments[idx]["speaker"] = f"spk_{int(lab):02d}"
    last = None
    for i, s in enumerate(segments):
        if "speaker" in s:
            last = s["speaker"]
        else:
            prev = next((segments[j]["speaker"] for j in range(i-1, -1, -1) if "speaker" in segments[j]), None)
            nxt  = next((segments[j]["speaker"] for j in range(i+1, len(segments)) if "speaker" in segments[j]), None)
            segments[i]["speaker"] = prev or nxt or (last or "spk_00")

def map_speakers_to_osoba(segments: List[Dict]) -> Dict[str, str]:
    order = []
    for s in segments:
        spk = s.get("speaker", "spk_00")
        if spk not in order:
            order.append(spk)
    return {spk: f"osoba{idx+1}" for idx, spk in enumerate(order)}

# =========================
# Transkrypcja + processing jednego nagrania
# =========================

def process_audio_file(audio_path: Path, cfg: Config) -> Dict:
    ffmpeg_bin = check_ffmpeg(cfg.ffmpeg_bin_env)
    tmp_dir = Path(cfg.tmp_dir); tmp_dir.mkdir(exist_ok=True)

    src = audio_path
    if cfg.denoise:
        src = ffmpeg_denoise_to_wav(ffmpeg_bin, audio_path, tmp_dir, cfg.ffmpeg_af)

    model = load_whisper(cfg.model, cfg.device, cfg.compute_type)
    hotwords_str = " ".join(cfg.hotwords or [])
    segments_iter, info = model.transcribe(
        str(src), language=cfg.language, task="transcribe",
        hotwords=hotwords_str if hotwords_str else None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=cfg.vad_min_silence_ms),
    )

    raw_segments = []
    for s in segments_iter:
        raw_segments.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": apply_name_fixes(s.text, cfg.proper_name_fixes).strip()
        })

    y, sr = sf.read(str(src), dtype="float32")
    if y.ndim > 1: y = y[:, 0]
    ambient = estimate_ambient_dbfs(y)

    if cfg.diarize and len(raw_segments) > 0:
        embs, sel_idx = segment_embeddings(y, sr, raw_segments, cfg)
        labels = cluster_speakers(embs, cfg.cluster_distance) if embs.size else []
        assign_cluster_labels(raw_segments, sel_idx, labels)
    else:
        for s in raw_segments:
            s["speaker"] = "spk_00"

    mapping = map_speakers_to_osoba(raw_segments)
    for s in raw_segments:
        s["speaker_human"] = mapping.get(s["speaker"], s["speaker"])

    for s in raw_segments:
        s_idx0 = int(s["start"] * sr)
        s_idx1 = int(s["end"] * sr)
        seg = y[max(0, s_idx0):min(len(y), s_idx1)]
        seg_db = dbfs(seg) if seg.size else ambient
        snr_db = seg_db - ambient
        f0_med, f0_std = estimate_pitch(seg, sr)
        words = max(1, len(s["text"].split()))
        dur = max(0.05, s["end"] - s["start"])
        wps = words / dur
        s["snr_db"] = float(snr_db)
        s["emotion"] = emotion_from_features(seg_db, f0_med, f0_std, wps)

    return {
        "file": audio_path.name,
        "ambient_dbfs": float(ambient),
        "segments": raw_segments
    }

# =========================
# Pretty print
# =========================

def print_segments(res: Dict):
    print(f"\n=== {res['file']} ===")
    print(f"[i] Poziom tła: {res['ambient_dbfs']:.1f} dBFS")
    for seg in res["segments"]:
        start = pretty_timestamp(seg["start"])
        end = pretty_timestamp(seg["end"])
        who = seg.get("speaker_human", seg.get("speaker", "osoba?"))
        snr = seg.get("snr_db", 0.0)
        emo = seg.get("emotion", "neutral")
        print(f'{who} [{start}–{end}] (+{snr:.1f} dB, {emo}): {seg["text"]}')

# =========================
# LOGS (jeden plik, nadpisywany)
# =========================

def init_logs(path: Path, mode: str):
    data = {
        "session_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "events": []
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def append_log(path: Path, event: Dict):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        doc = {"session_started": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "unknown", "events": []}
    doc["events"].append(event)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

# =========================
# BATCH
# =========================

def run_batch(cfg: Config):
    logs_path = Path(cfg.logs_path)
    init_logs(logs_path, "batch")

    in_dir = Path(cfg.in_dir)
    if not in_dir.exists():
        logging.error("Brak folderu: %s", in_dir.resolve())
        return
    files = [p for p in sorted(in_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        logging.warning("Brak plików w %s", in_dir.resolve())
        return

    print(f"Start batch, plików: {len(files)}")
    with ProcessPoolExecutor(max_workers=cfg.batch_workers) as ex:
        futs = {ex.submit(process_audio_file, f, cfg): f for f in files}
        for fut in as_completed(futs):
            try:
                res = fut.result()
                print_segments(res)
                append_log(logs_path, {"type": "file_result", "data": res})
                # eksporty (opcjonalnie)
                out_dir = Path("data_out"); out_dir.mkdir(exist_ok=True)
                stem = out_dir / Path(res["file"]).stem
                if "json" in cfg.export_formats:
                    (stem.with_suffix(".json")).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                if "csv" in cfg.export_formats:
                    import csv
                    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f, delimiter=";")
                        w.writerow(["speaker","start","end","snr_db","emotion","text"])
                        for s in res["segments"]:
                            w.writerow([s.get("speaker_human", s["speaker"]), f"{s['start']:.2f}", f"{s['end']:.2f}",
                                        f"{s.get('snr_db',0.0):.1f}", s.get("emotion","neutral"), s["text"]])
                if "srt" in cfg.export_formats:
                    def to_srt(t):
                        h=int(t//3600); m=int((t%3600)//60); s=t%60
                        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".",",")
                    with open(stem.with_suffix(".srt"),"w",encoding="utf-8") as f:
                        for i, s in enumerate(res["segments"], 1):
                            f.write(f"{i}\n{to_srt(s['start'])} --> {to_srt(s['end'])}\n")
                            f.write(f"{s.get('speaker_human', s['speaker'])}: {s['text']}\n\n")
                if "vtt" in cfg.export_formats:
                    def to_vtt(t):
                        h=int(t//3600); m=int((t%3600)//60); s=t%60
                        return f"{h:02d}:{m:02d}:{s:06.3f}"
                    with open(stem.with_suffix(".vtt"),"w",encoding="utf-8") as f:
                        f.write("WEBVTT\n\n")
                        for s in res["segments"]:
                            f.write(f"{to_vtt(s['start'])} --> {to_vtt(s['end'])}\n")
                            f.write(f"{s.get('speaker_human', s['speaker'])}: {s['text']}\n\n")
            except Exception:
                logging.exception("Błąd przetwarzania pliku")

# =========================
# LIVE (nasłuch + wake-word)
# =========================

class AudioRing:
    def __init__(self, max_seconds: float, sr: int):
        self.max_samples = int(max_seconds * sr)
        self.buffer = deque()
        self.size = 0
        self.sr = sr
    def push(self, chunk: np.ndarray):
        self.buffer.append(chunk.copy())
        self.size += len(chunk)
        while self.size > self.max_samples:
            old = self.buffer.popleft()
            self.size -= len(old)
    def dump(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self.buffer), axis=0)

def capture_and_process(buf: np.ndarray, sr: int, cfg: Config) -> Dict:
    tmp_dir = Path(cfg.tmp_dir); tmp_dir.mkdir(exist_ok=True)
    wav_path = tmp_dir / f"live_{int(time.time())}.wav"
    sf.write(str(wav_path), buf, sr)
    return process_audio_file(wav_path, cfg)

def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in t).lower()

def run_live(cfg: Config):
    logs_path = Path(cfg.logs_path)
    init_logs(logs_path, "live")

    sr = cfg.live_sample_rate
    frame = int(sr * (cfg.live_frame_ms / 1000.0))

    # VAD: 2 (bardziej agresywny -> łatwiej „cisza”)
    vad = webrtcvad.Vad(2)

    ring = AudioRing(cfg.live_max_pre_roll_sec, sr)
    probe_ring = AudioRing(2.0, sr)   # 2 s okno do sondowania

    ffmpeg_bin = check_ffmpeg(cfg.ffmpeg_bin_env)
    model = load_whisper(cfg.model, cfg.device, cfg.compute_type)

    hotwords_str = " ".join(cfg.hotwords or [])
    wake_phrases = [w.lower() for w in cfg.wake_phrases]
    wake_norm = [_normalize(w) for w in wake_phrases]

    input_kwargs = dict(channels=1, samplerate=sr, dtype="float32", blocksize=frame)
    if getattr(cfg, "live_input_device", None) is not None:
        input_kwargs["device"] = cfg.live_input_device

    stream = sd.InputStream(**input_kwargs)
    stream.start()
    print("Nasłuch… powiedz „Hej Watuś” lub „Hej Watusiu”")

    state = "IDLE"
    speech_started_ts = None
    last_probe_ts = 0.0
    consecutive_silence_ms = 0

    def energy_db(x: np.ndarray) -> float:
        if x.size == 0: return -120.0
        rms = np.sqrt(np.mean(x ** 2) + 1e-12)
        return 20 * np.log10(rms + 1e-12)

    try:
        while True:
            audio, _ = stream.read(frame)
            audio = audio.flatten().astype(np.float32)

            ring.push(audio)
            probe_ring.push(audio)

            # decyzja „czy cisza” na podstawie VAD + energii bieżącej ramki
            int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            vad_speech = vad.is_speech(int16, sr)
            frame_energy_db = energy_db(audio)

            if vad_speech or frame_energy_db > cfg.end_energy_dbfs:
                consecutive_silence_ms = 0
            else:
                consecutive_silence_ms += cfg.live_frame_ms

            now = time.time()

            # --- SONDOWANIE „WAKE WORD” (tylko w IDLE co ~0.6 s)
            if state == "IDLE" and (now - last_probe_ts) > 0.6:
                last_probe_ts = now
                bufp = probe_ring.dump()
                bufp = bufp[-int(2.0 * sr):]

                if len(bufp) > int(0.6 * sr) and energy_db(bufp) > cfg.wake_probe_min_dbfs:
                    tmp_dir = Path(cfg.tmp_dir); tmp_dir.mkdir(exist_ok=True)
                    raw_path = tmp_dir / "wake_tmp.wav"

                    bufp_boost = np.clip(bufp * 1.8, -1.0, 1.0)
                    sf.write(str(raw_path), bufp_boost, sr)
                    src = ffmpeg_denoise_to_wav(ffmpeg_bin, raw_path, tmp_dir, cfg.ffmpeg_af) if cfg.denoise else raw_path

                    segs, _ = model.transcribe(
                        str(src),
                        language=cfg.language,
                        task="transcribe",
                        hotwords=hotwords_str if hotwords_str else None,
                        vad_filter=False,
                        no_speech_threshold=0.30,
                        log_prob_threshold=-2.5,
                        beam_size=1,
                        condition_on_previous_text=False,
                    )
                    heard_text = " ".join(apply_name_fixes(s.text, cfg.proper_name_fixes) for s in segs).strip()
                    heard_norm = _normalize(heard_text)

                    # pokaż co usłyszała sonda, jeśli widzi „watu”
                    if "watu" in heard_norm:
                        print(f"[wake-probe] '{heard_text}'")

                    ok = any(w in heard_text.lower() for w in wake_phrases)
                    ok = ok or any(w in heard_norm for w in wake_norm)
                    ok = ok or ("hej" in heard_norm and "watus" in heard_norm)

                    if ok:
                        print("→ Wake-word rozpoznany. (ARMED)")
                        append_log(logs_path, {"type": "wake", "text": heard_text, "ts": time.time()})
                        state = "ARMED"
                        speech_started_ts = None
                        probe_ring = AudioRing(2.0, sr)
                        last_probe_ts = 0.0
                        consecutive_silence_ms = 0
                        continue

            # --- uzbrajanie i start
            if state == "ARMED":
                # start „capture”, gdy min. czas mowy po wywołaniu
                if speech_started_ts is None and (not vad_speech and consecutive_silence_ms == 0):
                    # wciąż cisza po wake — czekamy
                    pass
                if vad_speech and speech_started_ts is None:
                    speech_started_ts = now
                    print("→ Słyszę rozmówcę (CAPTURE).")
                if speech_started_ts and (now - speech_started_ts) * 1000 >= cfg.live_min_speech_ms:
                    state = "CAPTURE"

            # --- koniec wypowiedzi po ciszy skumulowanej
            if state in ("ARMED", "CAPTURE"):
                if consecutive_silence_ms >= max(cfg.live_silence_after_wake_ms, cfg.vad_min_silence_ms):
                    print("→ Koniec wypowiedzi, przetwarzam…")
                    buf = ring.dump()
                    res = capture_and_process(buf, sr, cfg)
                    print_segments(res)
                    append_log(logs_path, {"type": "live_result", "data": res, "ts": time.time()})

                    # wybór rozmówcy i echo TTS
                    by_spk = defaultdict(lambda: {"dur": 0.0, "snr": 0.0, "txt": []})
                    for m in res["segments"]:
                        who = m.get("speaker_human", m.get("speaker", "osoba?"))
                        dur = max(0.0, float(m["end"]) - float(m["start"]))
                        by_spk[who]["dur"] += dur
                        by_spk[who]["snr"] += max(0.0, float(m.get("snr_db", 0.0))) * dur
                        by_spk[who]["txt"].append(m["text"])
                    for k in by_spk:
                        d = by_spk[k]; d["score"] = d["dur"] + 0.03 * d["snr"]

                    leaders = sorted(by_spk.items(), key=lambda kv: kv[1]["score"], reverse=True)
                    if not leaders:
                        tts_say("Nie usłyszałem wypowiedzi.")
                    elif len(leaders) == 1 or (len(leaders) > 1 and leaders[0][1]["score"] > 1.25 * leaders[1][1]["score"]):
                        tts_say(" ".join(leaders[0][1]["txt"]).strip())
                    else:
                        parts = []
                        for name, d in leaders[:2]:
                            parts.append(f'{name}: {" ".join(d["txt"]).strip()}')
                        tts_say("; ".join(parts))

                    # reset
                    state = "IDLE"
                    ring = AudioRing(cfg.live_max_pre_roll_sec, sr)
                    probe_ring = AudioRing(2.0, sr)
                    last_probe_ts = 0.0
                    consecutive_silence_ms = 0
                    print("→ Gotowy (IDLE).")

    except KeyboardInterrupt:
        print("Zatrzymano (Ctrl+C).")
    except Exception:
        logging.exception("Błąd w trybie live")
    finally:
        try:
            stream.stop()
        except Exception:
            pass

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Watuś — batch & live (offline diarization, emotions)")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--mode", "-m", choices=["batch","live"], default="batch")
    parser.add_argument("--low-latency", action="store_true", help="Wymuś niską latencję (model=small, compute=int8)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.low_latency:
        cfg.model = "small"
        cfg.compute_type = "int8" if cfg.device == "cpu" else "int8_float16"

    setup_logging(cfg.log_level)

    Path(cfg.tmp_dir).mkdir(exist_ok=True)
    init_logs(Path(cfg.logs_path), args.mode)

    if args.mode == "batch":
        run_batch(cfg)
    else:
        run_live(cfg)

if __name__ == "__main__":
    main()
