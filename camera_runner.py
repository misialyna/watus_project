#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera → detections JSONL

- Domyślnie YOLO (ultralytics). Jeśli dostępne oczyWatusia_extracted/src/rtdetr.py
  i włączone --rt (lub VISION_USE_RTDETR=1), spróbuje RT-DETR i w razie problemów
  automatycznie przełączy się z powrotem na YOLO (bez crasha).

- Każdy wpis JSONL ma postać:
  {
    "ts": 1690000000.123,
    "objects": [{"name":"person","conf":0.91,"bbox":[x1,y1,x2,y2]}, ...],
    "brightness": 137.2,
    "size": [H, W]
  }

Uruchomienie (przykład):
  python camera_runner.py --jsonl "./camera.jsonl" --device 0 --rt

Zmienne środowiskowe (opcjonalne):
  CAMERA_JSONL         (ścieżka wyjścia JSONL)
  VISION_SCORE_THR     (0.5)
  VISION_WRITE_HZ      (5.0)
  VISION_USE_RTDETR    (0/1)
  RTDETR_WEIGHTS       (./oczyWatusia_extracted/rtdetr-l.pt)
"""

import os
import sys
import cv2
import time
import json
import argparse
import signal
from typing import List, Dict, Any, Optional

import numpy as np

from oczyWatusia import CVAgent

# ========= USTAWIENIA Z ENV =========
DEF_JSONL = os.environ.get("CAMERA_JSONL", "./camera.jsonl")
DEF_THR   = float(os.environ.get("VISION_SCORE_THR", "0.5"))
DEF_HZ    = float(os.environ.get("VISION_WRITE_HZ", "5.0"))
DEF_RT    = bool(int(os.environ.get("VISION_USE_RTDETR", "0")))

# NEW: główny katalog „oczyWatusia”
DEF_RTD_DIR = os.environ.get("RTDETR_DIR", "./oczyWatusia")
DEF_RTD_WEI = os.environ.get("RTDETR_WEIGHTS", os.path.join(DEF_RTD_DIR, "rtdetr-l.pt"))

# ========= YOLO =========
try:
    from ultralytics import YOLO
except Exception as e:
    print(f"[VISION] ultralytics import failed: {e}", flush=True)
    YOLO = None


class YoloDetector:
    def __init__(self, score_thr: float, model_path: Optional[str] = None):
        self.score_thr = float(score_thr)
        if model_path is None:
            # próbuj lokalny plik, inaczej pobierze automatycznie
            model_path = "yolov8n.pt"
        if YOLO is None:
            raise RuntimeError("ultralytics/YOLO not available")
        self.model = YOLO(model_path)

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        res = self.model.predict(source=frame_bgr, verbose=False)[0]
        out: List[Dict[str, Any]] = []
        names = res.names or {}
        for b, c, p in zip(res.boxes.xyxy.cpu().numpy(),
                           res.boxes.cls.cpu().numpy(),
                           res.boxes.conf.cpu().numpy()):
            if float(p) < self.score_thr:
                continue
            x1, y1, x2, y2 = map(int, b.tolist())
            cls = int(c)
            name = str(names.get(cls, f"cls_{cls}"))
            out.append({"name": name, "conf": float(p), "bbox": [x1, y1, x2, y2]})
        return out


# ========= RT-DETR (opcjonalnie) =========
class RTDetrDetector:
    """
    Wersja „Opcja A”:
      - dodajemy rodzica 'src' do sys.path
      - importujemy from src.rtdetr import RTDetrWrapper
      - miękki fallback na YOLO

    """
    def __init__(self, weights: str, score_thr: float):
        import os as _os, sys as _sys
        _sys.path.insert(0, _os.path.abspath(DEF_RTD_DIR))
        self._fallback = YoloDetector(score_thr=score_thr)
        self.core = None
        self.score_thr = score_thr
        try:
            from src.rtdetr import RTDetrWrapper  # type: ignore
            self.core = RTDetrWrapper(weights=weights, score_thresh=score_thr, device=None)
            print("[VISION] RT-DETR init OK", flush=True)
        except Exception as e:
            print(f"[VISION] RT-DETR init failed: {e} — fallback to YOLO", flush=True)
            self.core = None

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self.core is None:
            return self._fallback.detect(frame_bgr)
        try:
            dets = self.core.detect(frame_bgr) or []
        except Exception as e:
            print(f"[VISION] RT-DETR runtime error: {e} — fallback to YOLO", flush=True)
            return self._fallback.detect(frame_bgr)

        out: List[Dict[str, Any]] = []
        H, W = frame_bgr.shape[:2]
        for d in dets:
            name = str(d.get("label", "?"))
            score = float(d.get("score", 0.0))
            bbox  = d.get("bbox", [0, 0, 0, 0])
            try:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
                y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
                if score >= self.score_thr:
                    out.append({"name": name, "conf": score, "bbox": [x1, y1, x2, y2]})
            except Exception:
                continue
        return out


# ========= POMOCNICZE =========
def open_camera(device: str | int) -> cv2.VideoCapture:
    """
    device może być liczbą (index) albo nazwą (wtedy i tak próbujemy index 0).
    """
    idx: int
    if isinstance(device, int):
        idx = device
    else:
        s = str(device).strip()
        if s.isdigit():
            idx = int(s)
        else:
            # OpenCV i tak potrzebuje indeksu – zaczynamy od 0
            idx = 0
    cap = cv2.VideoCapture(idx)
    return cap


def write_jsonl(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            # f.close()
    except Exception as e:
        print(f"[VISION] JSONL write error: {e}", flush=True)


def brightness_of(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Camera to JSONL (YOLO / RT-DETR optional)")
    ap.add_argument("--jsonl", default=DEF_JSONL, help="Ścieżka do pliku JSONL (append)")
    ap.add_argument("--device", default=os.environ.get("CAMERA_DEVICE", "0"),
                    help="Index kamery lub nazwa; domyślnie 0")
    ap.add_argument("--rt", action="store_true" if DEF_RT else "store_false",
                    help="Wymuś RT-DETR (jeśli możliwe). Domyślnie wg ENV VISION_USE_RTDETR")
    ap.add_argument("--thr", type=float, default=DEF_THR, help="Próg pewności [0..1]")
    ap.add_argument("--model", default=os.environ.get("YOLO_MODEL", "yolov8n.pt"),
                    help="Ścieżka do modelu YOLO (opcjonalnie)")
    ap.add_argument("--hz", type=float, default=DEF_HZ, help="Częstotliwość zapisu JSONL (Hz)")
    ap.add_argument("--width", type=int, default=0, help="Wymuś szerokość (0=bez zmiany)")
    ap.add_argument("--height", type=int, default=0, help="Wymuś wysokość (0=bez zmiany)")
    ap.add_argument("--fps", type=int, default=0, help="Wymuś FPS kamery (0=bez zmiany)")
    return ap.parse_args()


# ========= GŁÓWNA PĘTLA =========
def main():
    args = parse_args()

    # Ctrl+C -> czyste wyjście
    def _sigint(*_):
        print("[VISION] stop", flush=True)
        try:
            cap.release()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    # Kamera
    cap = open_camera(args.device)
    agent = CVAgent(json_save_func=write_jsonl)
    agent.run(save_video=False, show_window=False)


if __name__ == "__main__":
    main()
