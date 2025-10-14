from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from ultralytics import YOLO

def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter + 1e-6)

def bbox_center(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5*(x1 + x2), 0.5*(y1 + y2))

def clamp_bbox(b: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = b
    x1 = min(max(0.0, x1), w - 1)
    x2 = min(max(0.0, x2), w - 1)
    y1 = min(max(0.0, y1), h - 1)
    y2 = min(max(0.0, y2), h - 1)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


@dataclass
class Track:
    track_id: int
    label: int
    bbox: List[float]              # xyxy
    score: float
    hits: int = 1                  # ile razy dopasowany
    age: int = 1                   # ile klatek istnienia
    time_since_update: int = 0     # ile klatek bez dopasowania
    vx: float = 0.0                # predykcja prędkości środka (px/klatkę)
    vy: float = 0.0
    last_center: Tuple[float, float

    ] = field(default_factory=lambda: (0.0, 0.0))

    def predict(self):
        # Ekstrapolacja prostoliniowa środka; rozmiar bboxa zostawiamy.
        cx, cy = bbox_center(self.bbox)
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        cx_pred = cx + self.vx
        cy_pred = cy + self.vy
        x1 = cx_pred - w/2
        y1 = cy_pred - h/2
        x2 = cx_pred + w/2
        y2 = cy_pred + h/2
        self.bbox = [x1, y1, x2, y2]
        self.age += 1
        self.time_since_update += 1

    def update(self, det_bbox: List[float], det_score: float):
        # Uaktualnienie pozycji i prędkości (prosta różnica środków)
        cx_prev, cy_prev = bbox_center(self.bbox)
        cx_new, cy_new = bbox_center(det_bbox)
        self.vx = cx_new - cx_prev
        self.vy = cy_new - cy_prev
        self.bbox = det_bbox
        self.score = det_score
        self.hits += 1
        self.time_since_update = 0
        self.last_center = (cx_new, cy_new)


class IoUTracker:
    def __init__(
        self,
        iou_thresh: float = 0.35,
        score_thresh: float = 0.4,
        max_age: int = 10,
        min_hits: int = 6
    ):
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: List[Track] = []

    def _match(self, dets: List[Dict]) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        """
        Greedy: sortujemy pary po IoU malejąco, matchujemy niekonfliktowe z IoU>=thr i taką samą klasą.
        Zwraca: matches (track_idx, det_idx), unmatched_tracks, unmatched_dets
        """
        if not self.tracks or not dets:
            return [], list(range(len(self.tracks))), list(range(len(dets)))

        # Macierz par (track_i, det_j, iou), tylko gdy etykiety się zgadzają.
        pairs = []
        for ti, t in enumerate(self.tracks):
            for dj, d in enumerate(dets):
                if d["score"] < self.score_thresh:
                    continue
                if d["label"] != t.label:
                    continue
                iou = iou_xyxy(t.bbox, d["bbox"])
                if iou >= self.iou_thresh:
                    pairs.append((ti, dj, iou))
        pairs.sort(key=lambda x: x[2], reverse=True)

        used_t, used_d = set(), set()
        matches: List[Tuple[int,int]] = []
        for ti, dj, _ in pairs:
            if ti in used_t or dj in used_d:
                continue
            used_t.add(ti)
            used_d.add(dj)
            matches.append((ti, dj))

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_t]
        unmatched_dets = [j for j in range(len(dets)) if j not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    def step(self, frame, dets: List[Dict]) -> List[Dict]:
        """
        dets: lista słowników {bbox:[x1,y1,x2,y2], score:float, label:int}
        frame_size: (W,H) – by klampować bboxy po predykcji
        Zwraca listę torów do wizualizacji: {track_id, label, bbox, score, confirmed}
        """
        W, H = len(frame[0]), len(frame)

        # 1) Predykcja pozycji istniejących torów
        for t in self.tracks:
            t.predict()
            t.bbox = clamp_bbox(t.bbox, W, H)

        # 2) Dopasowanie
        matches, unmatched_tracks, unmatched_dets = self._match(dets)

        # 3) Update dopasowanych
        for ti, dj in matches:
            det = dets[dj]
            self.tracks[ti].update(det["bbox"], det["score"])
            self.tracks[ti].bbox = clamp_bbox(self.tracks[ti].bbox, W, H)

        # 4) Tworzenie nowych torów z niedopasowanych detekcji
        for dj in unmatched_dets:
            d = dets[dj]
            if d["score"] < self.score_thresh:
                continue
            t = Track(
                track_id=self.next_id,
                label=d["label"],
                bbox=clamp_bbox(d["bbox"], W, H),
                score=d["score"],
            )
            t.last_center = bbox_center(t.bbox)
            self.tracks.append(t)
            self.next_id += 1

        # 5) Usuwanie torów, które dawno nie były aktualizowane
        alive: List[Track] = []
        for t in self.tracks:
            if t.time_since_update <= self.max_age:
                alive.append(t)
        self.tracks = alive

        # 6) Wynik dla wizualizacji / dalszego przetwarzania
        outputs = []
        for t in self.tracks:
            outputs.append({
                "track_id": t.track_id,
                "label": t.label,
                "bbox": t.bbox,
                "score": t.score,
                "confirmed": t.hits >= self.min_hits
            })
        return outputs
