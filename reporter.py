#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import re
import threading
from collections import deque, Counter
from typing import Set, Dict, Any, Optional, Tuple

import zmq
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

# ===== ENV =====
load_dotenv()
ZMQ_PUB_ADDR = os.environ.get("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")
ZMQ_SUB_ADDR = os.environ.get("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")

LLM_HTTP_URL = (os.environ.get("LLM_HTTP_URL") or "").strip()
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", os.environ.get("LLM_HTTP_TIMEOUT", "30")))

SCENARIOS_DIR = os.environ.get("WATUS_SCENARIOS_DIR", "./scenarios_text")
SCENARIO_ACTIVE_PATH = os.environ.get("SCENARIO_ACTIVE_PATH", os.path.join(SCENARIOS_DIR, "active.jsonl"))

CAMERA_NAME  = os.environ.get("CAMERA_NAME", "cam_front")
CAMERA_JSONL = os.environ.get("CAMERA_JSONL", "/Users/michalinamoszynska/Documents/GitHub/watus_project/camera.jsonl")

LOG_DIR   = os.environ.get("LOG_DIR", "./")
RESP_FILE = os.path.join(LOG_DIR, "responses.jsonl")
MELD_FILE = os.path.join(LOG_DIR, "meldunki.jsonl")

# ===== ZMQ =====
ctx = zmq.Context.instance()

sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "dialog.leader")
sub.connect(ZMQ_PUB_ADDR)

pub = ctx.socket(zmq.PUB)
pub.setsockopt(zmq.SNDHWM, 100)
pub.setsockopt(zmq.LINGER, 0)
pub.bind(ZMQ_SUB_ADDR)

# ===== HTTP API =====
app = FastAPI()

@app.on_event("startup")
def _startup():
    print(f"[Reporter] SUB dialog.leader  @ {ZMQ_PUB_ADDR}", flush=True)
    print(f"[Reporter] PUB tts.speak      @ {ZMQ_SUB_ADDR}", flush=True)
    print(f"[Reporter] LLM_HTTP_URL       = {LLM_HTTP_URL or '(BRAK)'}  timeout={HTTP_TIMEOUT:.1f}s", flush=True)
    print(f"[Reporter] CAMERA_JSONL       = {CAMERA_JSONL or '(OFF)'}", flush=True)
    print(f"[Reporter] SCENARIO_ACTIVE    = {SCENARIO_ACTIVE_PATH}", flush=True)

@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": time.time(),
        "llm_url": LLM_HTTP_URL,
        "scenario": get_active_scenario(),
        "camera_tail_active": _camera_last.get("ts", 0.0) > 0.0
    }

# ===== Duplikaty =====
_seen_turn_ids: Set[int] = set()
_SEEN_LIMIT = 10000

def seen(turn_ids) -> bool:
    if not turn_ids: return False
    try: tid = int(turn_ids[0])
    except Exception: return False
    if tid in _seen_turn_ids: return True
    _seen_turn_ids.add(tid)
    if len(_seen_turn_ids) > _SEEN_LIMIT:
        for _ in range(len(_seen_turn_ids)//2):
            try: _seen_turn_ids.pop()
            except KeyError: break
    return False

# ===== Pliki JSONL =====
def write_jsonl(path: str, obj: Dict[str, Any]):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ===== Scenariusz (watch) =====
_sc_lock = threading.Lock()
_active_scenario = os.environ.get("SCENARIO", "default")

def _read_active_scenario(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            last = None
            for line in f:
                last = line
            if not last:
                return _active_scenario
            obj = json.loads(last.strip())
            sid = str(obj.get("id") or "").strip()
            return sid if sid else _active_scenario
    except Exception:
        return _active_scenario

def scenario_watch_loop(path: str, poll_s: float = 1.0):
    global _active_scenario
    prev = None
    while True:
        sid = _read_active_scenario(path)
        if sid and sid != prev:
            with _sc_lock:
                _active_scenario = sid
            print(f"[Reporter] Aktywny scenariusz → {sid}", flush=True)
            prev = sid
        time.sleep(poll_s)

def get_active_scenario() -> str:
    with _sc_lock:
        return _active_scenario

# ===== Kamera: tail + okno agregacji =====
# Buforujemy ostatnie ~2.5s dla stabilniejszego meldunku
_cam_lock = threading.Lock()
_camera_last: Dict[str, Any] = {"ts": 0.0, "record": None}
_camera_buf: deque[Dict[str, Any]] = deque()
CAM_WINDOW_SEC = float(os.environ.get("CAMERA_WINDOW_SEC", "2.5"))

def _summarize_frame(obj: Dict[str, Any]) -> str:
    items = obj.get("objects") or obj.get("detections") or []
    if not items:
        br = obj.get("brightness")
        br_s = f" | bright={br:.2f}" if isinstance(br, (int,float)) else ""
        return f"brak detekcji{br_s}".strip()
    items = sorted(items, key=lambda x: float(x.get("conf", 0.0)), reverse=True)
    top = [f"{(it.get('name') or '?')}({int(round(100*float(it.get('conf',0.0))))}%)" for it in items[:3]]
    n = len(items)
    extra = f"+{n-3}" if n > 3 else ""
    br = obj.get("brightness")
    br_s = f" | bright={br:.2f}" if isinstance(br, (int,float)) else ""
    return f"{', '.join(top)} {extra}{br_s}".strip()

def _summarize_window(buf: deque) -> Dict[str, Any]:
    # liczymy najczęstsze obiekty oraz średnią jasność
    counts = Counter()
    br_sum = 0.0
    br_n = 0
    last_ts = 0.0
    for rec in buf:
        items = rec.get("objects") or []
        for it in items:
            name = str(it.get("name") or "?")
            counts[name] += 1
        br = rec.get("brightness")
        if isinstance(br, (int, float)):
            br_sum += float(br); br_n += 1
        last_ts = max(last_ts, float(rec.get("ts") or 0.0))
    top = [{"name": n, "count": c} for n, c in counts.most_common(3)]
    avg_bri = (br_sum/br_n) if br_n > 0 else None
    return {
        "since": CAM_WINDOW_SEC,
        "top_objects": top,
        "avg_brightness": avg_bri,
        "last_ts": last_ts,
    }

def camera_tail_loop(path: str):
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.05); continue
                try:
                    obj = json.loads(line.strip())
                    ts = float(obj.get("ts") or time.time())
                    with _cam_lock:
                        _camera_last["ts"] = ts
                        _camera_last["record"] = obj
                        _camera_buf.append(obj)
                        # trim do okna czasowego
                        cutoff = ts - CAM_WINDOW_SEC
                        while _camera_buf and float(_camera_buf[0].get("ts", 0.0)) < cutoff:
                            _camera_buf.popleft()
                except Exception:
                    pass
    except Exception as e:
        print(f"[Reporter][CAM] tail err: {e}", flush=True)

# ===== Meldunek =====
def build_meldunek(msg: Dict[str, Any]) -> Dict[str, Any]:
    question   = (msg.get("text_full") or "").strip()
    session_id = msg.get("session_id")
    group_id   = msg.get("group_id")
    ts_start   = float(msg.get("ts_start") or 0.0)
    ts_end     = float(msg.get("ts_end") or 0.0)
    dbfs       = msg.get("dbfs")
    verify     = msg.get("verify") or {}
    now        = time.time()
    scenario_id = get_active_scenario()

    # kamera – ostatnia klatka + okno
    with _cam_lock:
        last_rec = _camera_last.get("record")
        last_summary = _summarize_frame(last_rec) if last_rec else "brak danych"
        win = _summarize_window(_camera_buf) if _camera_buf else None

    # krótkie stringi w opisie (zostawiamy szybki POST tylko ze stringiem)
    cam_last_s = f"LAST={last_summary}" if last_rec else "LAST=none"
    if win:
        top_str = ", ".join([f"{t['name']}×{t['count']}" for t in (win.get('top_objects') or [])]) or "none"
        if isinstance(win.get("avg_brightness"), (int,float)):
            cam_win_s = f"WIN{int(CAM_WINDOW_SEC*1000)}ms: {top_str} | avg_bri={win['avg_brightness']:.2f}"
        else:
            cam_win_s = f"WIN{int(CAM_WINDOW_SEC*1000)}ms: {top_str}"
    else:
        cam_win_s = "WIN=none"

    opis = (
        f"[SYS_TIME={now:.3f}] [SCENARIO={scenario_id}] [CAMERA={CAMERA_NAME}] "
        f"[SESSION={session_id}] [GROUP={group_id}] "
        f"[SPEECH={ts_start:.3f}-{ts_end:.3f}s ~{dbfs:.1f}dBFS] "
        f"[LEADER_SCORE={verify.get('score')}] "
        f"[VISION {cam_last_s} | {cam_win_s}] "
        f"USER: {question}"
    )

    # pełny meldunek do logów/debug (do JSONL) – nie wysyłamy go do LLM (zachowujemy kompatybilność)
    meld_json = {
        "ts_system": now,
        "scenario": scenario_id,
        "camera": {"name": CAMERA_NAME, "jsonl_path": CAMERA_JSONL or None},
        "question_text": question,
        "opis": opis,
        "dialog_meta": {
            "session_id": session_id,
            "group_id": group_id,
            "turn_ids": msg.get("turn_ids"),
            "ts_start": ts_start,
            "ts_end": ts_end,
            "dbfs": dbfs,
            "verify": verify,
        },
        "vision": {
            "last": last_rec,
            "window_summary": win
        }
    }
    return meld_json

def print_meldunek(m: Dict[str, Any]):
    print(
        "\n[Reporter][MELDUNEK]"
        f"\n- ts_system : {m['ts_system']:.3f}"
        f"\n- scenariusz: {m['scenario']}"
        f"\n- kamera    : {m['camera']['name']}"
        f"\n- opis→LLM  : {m['opis']}\n",
        flush=True
    )

# ===== HTTP (LLM) =====
_RETRY_IN_RE = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_retry_hint(err_body: str) -> Optional[float]:
    if not err_body: return None
    if ("RESOURCE_EXHAUSTED" in err_body or " 429 " in err_body or "\"code\": 429" in err_body):
        m = _RETRY_IN_RE.search(err_body)
        if m:
            try: return max(1.0, min(float(m.group(1)), 60.0))
            except Exception: return 5.0
        return 5.0
    return None

def ask_llm_string(content_text: str) -> Tuple[Optional[str], Optional[int], Optional[str], float]:
    """
    Zostawiamy szybki POST z jednym polem 'content'.
    Zwraca: (answer, status_code, error_text, latency_ms)
    """
    if not LLM_HTTP_URL:
        return None, None, "LLM_HTTP_URL is empty", 0.0

    body = json.dumps({"content": content_text}, ensure_ascii=True).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}

    t0 = time.time()
    try:
        print(f"[Reporter][HTTP→] POST {LLM_HTTP_URL} len={len(content_text)}", flush=True)
        r = requests.post(LLM_HTTP_URL, data=body, headers=headers, timeout=HTTP_TIMEOUT)
        latency = (time.time() - t0) * 1000.0
        print(f"[Reporter][HTTP←] {r.status_code} ({latency:.1f} ms)", flush=True)

        if 200 <= r.status_code < 300:
            try:
                data = r.json()
            except Exception:
                data = {"raw_text": r.text}
            ans = (data.get("answer") or data.get("msg") or data.get("text") or "").strip()
            write_jsonl(RESP_FILE, {
                "ts": time.time(),
                "request": content_text,
                "raw_response": data,
                "answer": ans,
                "latency_ms": latency
            })
            return (ans if ans else json.dumps(data, ensure_ascii=False)), r.status_code, None, latency

        err_body = r.text
        print(f"[Reporter][HTTP!] status={r.status_code} body={err_body[:400]}", flush=True)
        return None, r.status_code, err_body, latency

    except requests.Timeout as e:
        return None, 408, f"timeout: {e}", (time.time() - t0) * 1000.0
    except requests.RequestException as e:
        return None, None, f"request_exception: {e}", (time.time() - t0) * 1000.0

# ===== Pętla główna =====
def loop():
    time.sleep(0.2)
    while True:
        try:
            topic, payload = sub.recv_multipart()
            if topic != b"dialog.leader":
                continue

            try:
                msg = json.loads(payload.decode("utf-8"))
            except Exception:
                continue

            turn_ids = msg.get("turn_ids") or []
            group_id = msg.get("group_id")

            if seen(turn_ids):
                print(f"[Reporter][RECV] dup turn_id={turn_ids[0]} – pomijam", flush=True)
                continue

            meld = build_meldunek(msg)
            print_meldunek(meld)
            write_jsonl(MELD_FILE, meld)

            content_text = meld["opis"]  # tylko string – kompatybilność i szybkość

            # 1. próba
            ans, status, err, lat_ms = ask_llm_string(content_text)

            # ewentualny retry (rzadko potrzebny – POST i tak jest szybki)
            wait_hint = parse_retry_hint(err or "") if not ans and status == 500 else None
            retried = False
            if not ans:
                if wait_hint is not None:
                    print(f"[Reporter][HTTP] backend 500/429 – czekam {wait_hint:.1f}s i retry", flush=True)
                    time.sleep(wait_hint); retried = True
                    ans, status, err, _ = ask_llm_string(content_text)
                elif status in {408, 429, 502, 503, 504, None}:
                    import random
                    backoff = 0.40 + random.random() * 0.50
                    print(f"[Reporter][HTTP] retryable status={status} – backoff {backoff:.2f}s", flush=True)
                    time.sleep(backoff); retried = True
                    ans, status, err, _ = ask_llm_string(content_text)

            if not ans:
                msg_text = "Przepraszam, serwer odpowiedzi jest chwilowo przeciążony. Spróbuj proszę za moment."
                print(f"[Reporter][DROP] brak odpowiedzi po {'retry' if retried else '1. próbie'} (status={status})", flush=True)
                pub.send_multipart([b"tts.speak", json.dumps(
                    {"text": msg_text, "reply_to": group_id, "turn_ids": turn_ids},
                    ensure_ascii=False).encode("utf-8")])
                continue

            print(f"[Reporter][LLM] answer len={len(ans)}", flush=True)
            pub.send_multipart([b"tts.speak", json.dumps(
                {"text": ans, "reply_to": group_id, "turn_ids": turn_ids},
                ensure_ascii=False).encode("utf-8")])
            print(f"[Reporter][PUB] tts.speak → reply_to={group_id} len={len(ans)}", flush=True)

        except Exception as e:
            print(f"[Reporter] loop exception: {e}", flush=True)
            time.sleep(0.15)

# ===== Main =====
def main():
    if CAMERA_JSONL:
        threading.Thread(target=camera_tail_loop, args=(CAMERA_JSONL,), daemon=True).start()
    if SCENARIO_ACTIVE_PATH:
        threading.Thread(target=scenario_watch_loop, args=(SCENARIO_ACTIVE_PATH,), daemon=True).start()
    thr = threading.Thread(target=loop, daemon=True)
    thr.start()
    uvicorn.run(app, host="127.0.0.1", port=8781, log_level="info")

if __name__ == "__main__":
    main()
