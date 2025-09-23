#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reporter:
- SUB 'dialog.leader'  (connect do ZMQ_PUB_ADDR → 7780)
- Składa „meldunek” (opis + metadane), DRUKUJE go w terminalu i zapisuje
- Wysyła do LLM **meldunek jako jeden string** (kontrakt: {'content': '<string>'})
- Jeżeli backend zwraca 500, ale w treści jest 429/RESOURCE_EXHAUSTED + "retry in X",
  to czekamy X sekund (w granicach) i retry'ujemy raz.
- Przy braku odpowiedzi po retrze – wysyłamy krótki TTS z komunikatem do użytkownika.
- PUB 'tts.speak'      (bind na ZMQ_SUB_ADDR → 7781)
- Loguje odpowiedzi LLM do responses.jsonl oraz meldunki do meldunki.jsonl
"""

import os
import time
import json
import re
import threading
from typing import Set, Dict, Any, Optional, Tuple

import zmq
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

load_dotenv()

ZMQ_PUB_ADDR = os.environ.get("ZMQ_PUB_ADDR", "tcp://127.0.0.1:7780")
ZMQ_SUB_ADDR = os.environ.get("ZMQ_SUB_ADDR", "tcp://127.0.0.1:7781")

LLM_HTTP_URL = (os.environ.get("LLM_HTTP_URL") or "").strip()  # np. http://127.0.0.1:8000/api1/process_question
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", os.environ.get("LLM_HTTP_TIMEOUT", "30")))

SCENARIO     = os.environ.get("SCENARIO", "default")
CAMERA_NAME  = os.environ.get("CAMERA_NAME", "cam_front")
CAMERA_JSONL = os.environ.get("CAMERA_JSONL", "")

LOG_DIR      = os.environ.get("LOG_DIR", "./")
RESP_FILE    = os.path.join(LOG_DIR, "responses.jsonl")
MELD_FILE    = os.path.join(LOG_DIR, "meldunki.jsonl")

# ZMQ
ctx = zmq.Context.instance()

sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, "dialog.leader")
sub.connect(ZMQ_PUB_ADDR)

pub = ctx.socket(zmq.PUB)
pub.setsockopt(zmq.SNDHWM, 100)
pub.setsockopt(zmq.LINGER, 0)
pub.bind(ZMQ_SUB_ADDR)

app = FastAPI()

@app.on_event("startup")
def _startup():
    print(f"[Reporter] SUB dialog.leader  @ {ZMQ_PUB_ADDR}", flush=True)
    print(f"[Reporter] PUB tts.speak      @ {ZMQ_SUB_ADDR}", flush=True)
    print(f"[Reporter] LLM_HTTP_URL       = {LLM_HTTP_URL or '(BRAK)'}  timeout={HTTP_TIMEOUT}s", flush=True)

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time(), "llm_url": LLM_HTTP_URL, "scenario": SCENARIO}

# deduplikacja
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

def write_jsonl(path: str, obj: Dict[str, Any]):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---- Meldunek ----
def build_meldunek(msg: Dict[str, Any]) -> Dict[str, Any]:
    question  = (msg.get("text_full") or "").strip()
    session_id= msg.get("session_id")
    group_id  = msg.get("group_id")
    ts_start  = float(msg.get("ts_start") or 0.0)
    ts_end    = float(msg.get("ts_end") or 0.0)
    dbfs      = msg.get("dbfs")
    verify    = msg.get("verify") or {}
    now       = time.time()

    opis = (
        f"[SYS_TIME={now:.3f}] "
        f"[SCENARIO={SCENARIO}] "
        f"[CAMERA={CAMERA_NAME}] "
        f"[SESSION={session_id}] [GROUP={group_id}] "
        f"[SPEECH={ts_start:.3f}-{ts_end:.3f}s ~{dbfs:.1f}dBFS] "
        f"[LEADER_SCORE={verify.get('score')}] "
        f"USER: {question}"
    )

    meld = {
        "ts_system": now,
        "scenario": SCENARIO,
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
    }
    return meld

def print_meldunek(m: Dict[str, Any]):
    print(
        "\n[Reporter][MELDUNEK]"
        f"\n- ts_system : {m['ts_system']:.3f}"
        f"\n- scenariusz: {m['scenario']}"
        f"\n- kamera    : {m['camera']['name']}"
        f"\n- opis→LLM  : {m['opis']}\n",
        flush=True
    )

# ---- HTTP ----
def ask_llm_string(content_text: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Zwraca: (answer|None, http_status|None, err_body|None)"""
    if not LLM_HTTP_URL:
        return None, None, "LLM_HTTP_URL is empty"

    payload = {"content": content_text}
    try:
        print(f"[Reporter][HTTP→] POST {LLM_HTTP_URL} len={len(content_text)}", flush=True)
        r = requests.post(LLM_HTTP_URL, json=payload, timeout=HTTP_TIMEOUT)
        print(f"[Reporter][HTTP←] {r.status_code}", flush=True)

        if 200 <= r.status_code < 300:
            try:
                data = r.json()
            except Exception:
                data = {"raw_text": r.text}
            ans = (data.get("answer") or data.get("msg") or data.get("text") or "").strip()
            write_jsonl(RESP_FILE, {"ts": time.time(), "request": content_text, "raw_response": data, "answer": ans})
            return (ans if ans else json.dumps(data, ensure_ascii=False)), r.status_code, None

        err_body = r.text
        print(f"[Reporter][HTTP!] status={r.status_code} body={err_body[:400]}", flush=True)
        return None, r.status_code, err_body

    except requests.Timeout as e:
        return None, 408, f"timeout: {e}"
    except requests.RequestException as e:
        return None, None, f"request_exception: {e}"

# rozpoznanie "500, ale w treści jest 429 + retry in N"
_RETRY_IN_RE = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
def parse_retry_hint(err_body: str) -> Optional[float]:
    if not err_body: return None
    if ("RESOURCE_EXHAUSTED" in err_body or " 429 " in err_body or "\"code\": 429" in err_body):
        m = _RETRY_IN_RE.search(err_body)
        if m:
            try:
                secs = float(m.group(1))
                return max(1.0, min(secs, 60.0))  # clamp
            except Exception:
                return 5.0
        return 5.0
    return None

def loop():
    time.sleep(0.2)  # krótszy warm-up

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

            # meldunek
            meld = build_meldunek(msg)
            print_meldunek(meld)
            write_jsonl(MELD_FILE, meld)
            content_text = meld["opis"]

            # 1. próba
            ans, status, err = ask_llm_string(content_text)

            # Specjalny przypadek: 500, ale w treści 429 + "retry in X"
            wait_hint = None
            if not ans and status == 500 and isinstance(err, str):
                wait_hint = parse_retry_hint(err)

            retried = False
            if not ans:
                if wait_hint is not None:
                    print(f"[Reporter][HTTP] backend 500/429 – czekam {wait_hint:.1f}s i retry", flush=True)
                    time.sleep(wait_hint)
                    ans, status, err = ask_llm_string(content_text)
                    retried = True
                else:
                    # standardowe retriable
                    retryable = {408, 429, 502, 503, 504, None}
                    if status in retryable:
                        import random
                        delay = 0.45 + random.random() * 0.5
                        print(f"[Reporter][HTTP] retryable status={status} – backoff {delay:.2f}s", flush=True)
                        time.sleep(delay)
                        ans, status, err = ask_llm_string(content_text)
                        retried = True

            if not ans:
                # Fallback TTS (krótka informacja) – żeby Watus zasygnalizował co się dzieje
                msg_text = "Przepraszam, serwer odpowiedzi jest chwilowo przeciążony. Spróbuj proszę za moment."
                print(f"[Reporter][DROP] brak odpowiedzi po {'retry' if retried else '1. próbie'} (status={status})", flush=True)
                out = {"text": msg_text, "reply_to": group_id, "turn_ids": turn_ids}
                pub.send_multipart([b"tts.speak", json.dumps(out, ensure_ascii=False).encode("utf-8")])
                continue

            # pokaż odpowiedź LLM w reporterze
            print(f"[Reporter][LLM] answer len={len(ans)}", flush=True)

            out = {"text": ans, "reply_to": group_id, "turn_ids": turn_ids}
            pub.send_multipart([b"tts.speak", json.dumps(out, ensure_ascii=False).encode("utf-8")])

            print(f"[Reporter][PUB] tts.speak → reply_to={group_id} len={len(ans)}", flush=True)

        except Exception as e:
            print(f"[Reporter] loop exception: {e}", flush=True)
            time.sleep(0.15)

def main():
    thr = threading.Thread(target=loop, daemon=True)
    thr.start()
    uvicorn.run(app, host="127.0.0.1", port=8781, log_level="info")

if __name__ == "__main__":
    main()
