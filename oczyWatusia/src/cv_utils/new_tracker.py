from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import time
import math
import numpy as np
import cv2
import torch

# ========== 0) Konfiguracja domyślna ==========
class Cfg:
    # RT-DETR
    weights       = "rtdetr-l.pt"   # rtdetr-r18.pt (szybki) / rtdetr-l.pt / rtdetr-x.pt
    imgsz         = 640          # większe -> lepsze dla małych obiektów (GPU!)
    conf_det      = 0.35              # próg detektora
    iou_nms       = 0.55              # NMS IoU
    max_det       = 300

    # Tracker
    iou_gate      = 0.20              # minimalny IoU do rozważenia pary
    mhal_gate     = 9.21              # ~chi^2(4) @ p≈0.01
    alpha_cost    = 0.60              # waga Mahalanobisa vs IoU: cost = α*Mhal+(1-α)*(1-IoU)
    max_age       = 30                # ile klatek bez update zanim usuniemy tor
    min_hits      = 4                 # ile update'ów zanim tor "confirmed"
    ema_alpha     = 0.70              # smoothing: im większe, tym gładsze

    # Aparycja (opcjonalnie)
    use_appearance = False
    app_bins       = (8, 8, 4)        # HSV histogram
    app_beta       = 0.20             # waga kosztu aparycji (cosine dist)
    app_gate       = 0.60             # dopuszczalny dystans kosinusowy

    # Kalman (strojenie szumu)
    R_pos          = 2.0              # wariancja pomiaru cx,cy
    R_s            = 10.0             # wariancja pomiaru s
    R_r            = 10.0             # wariancja pomiaru r
    Q_cxcy         = 1e-3
    Q_s            = 1e-2
    Q_vel          = 1e-1

# ========== 1) RT-DETR wrapper (Ultralytics) ==========
class RTDetrWrapper:
    def __init__(self, weights=Cfg.weights, score_thresh=Cfg.conf_det, device: Optional[str]=None,
                 imgsz=Cfg.imgsz, iou_nms=Cfg.iou_nms, max_det=Cfg.max_det):
        from ultralytics import YOLO
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights)
        self.model.to(self.device)
        self.conf = score_thresh
        self.imgsz = imgsz
        self.iou_nms = iou_nms
        self.max_det = max_det
        self.class_names = self.model.names

    @torch.inference_mode()
    def __call__(self, frame_bgr: np.ndarray) -> List[Dict]:
        res = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou_nms,
            max_det=self.max_det,
            device=self.device,
            verbose=False
        )[0]
        dets: List[Dict] = []
        if res.boxes is None or res.boxes.shape[0] == 0:
            return dets
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        conf = res.boxes.conf.detach().cpu().numpy()
        cls  = res.boxes.cls.detach().cpu().numpy().astype(int)
        for b, s, l in zip(xyxy, conf, cls):
            dets.append({"bbox":[float(b[0]),float(b[1]),float(b[2]),float(b[3])],
                         "score":float(s), "label":int(l)})
        return dets

# ========== 2) Narzędzia bbox ==========
def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0: return 0.0
    ar = (ax2-ax1)*(ay2-ay1); br = (bx2-bx1)*(by2-by1)
    return inter / (ar + br - inter + 1e-6)

def clamp_bbox(b: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = b
    x1 = min(max(0.0, x1), w - 1); x2 = min(max(0.0, x2), w - 1)
    y1 = min(max(0.0, y1), h - 1); y2 = min(max(0.0, y2), h - 1)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def nms_per_class(dets, iou_thr=0.5):
    """
    dets: list[{bbox:[x1,y1,x2,y2], score, label}]
    Zwraca zredukowaną listę po NMS (per klasa).
    """
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d["label"], []).append(d)
    out = []
    for cls, arr in by_cls.items():
        # sort by score desc
        arr = sorted(arr, key=lambda x: x["score"], reverse=True)
        keep = []
        while arr:
            best = arr.pop(0)
            keep.append(best)
            arr = [x for x in arr if iou_xyxy(best["bbox"], x["bbox"]) < iou_thr]
        out.extend(keep)
    return out

# ========== 3) Kalman Box (SORT-style) ==========
class KalmanBox:
    """
    Stan: x = [cx, cy, s, r, vx, vy, vs]^T
    Pomiar: z = [cx, cy, s, r]
    """
    def __init__(self, init_bbox_xyxy: List[float], dt: float=1.0):
        self.dt = dt
        cx = 0.5*(init_bbox_xyxy[0] + init_bbox_xyxy[2])
        cy = 0.5*(init_bbox_xyxy[1] + init_bbox_xyxy[3])
        w  = max(1e-3, init_bbox_xyxy[2]-init_bbox_xyxy[0])
        h  = max(1e-3, init_bbox_xyxy[3]-init_bbox_xyxy[1])
        s  = w*h; r = w/h

        self.x = np.array([cx, cy, s, r, 0., 0., 0.], dtype=float)

        self.F = np.eye(7)
        self.F[0,4] = dt; self.F[1,5] = dt; self.F[2,6] = dt

        self.H = np.zeros((4,7))
        self.H[0,0]=1.; self.H[1,1]=1.; self.H[2,2]=1.; self.H[3,3]=1.

        self.P = np.eye(7) * 10.0
        self.R = np.diag([Cfg.R_pos, Cfg.R_pos, Cfg.R_s, Cfg.R_r])
        self.Q = np.diag([Cfg.Q_cxcy, Cfg.Q_cxcy, Cfg.Q_s, 1e-3, Cfg.Q_vel, Cfg.Q_vel, Cfg.Q_vel])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.to_xyxy()

    def update(self, det_bbox_xyxy: List[float]):
        cx = 0.5*(det_bbox_xyxy[0] + det_bbox_xyxy[2])
        cy = 0.5*(det_bbox_xyxy[1] + det_bbox_xyxy[3])
        w  = max(1e-3, det_bbox_xyxy[2]-det_bbox_xyxy[0])
        h  = max(1e-3, det_bbox_xyxy[3]-det_bbox_xyxy[1])
        s  = w*h; r = w/h
        z = np.array([cx, cy, s, r], dtype=float)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def gating_distance(self, det_bbox_xyxy: List[float]) -> float:
        cx = 0.5*(det_bbox_xyxy[0] + det_bbox_xyxy[2])
        cy = 0.5*(det_bbox_xyxy[1] + det_bbox_xyxy[3])
        w  = max(1e-3, det_bbox_xyxy[2]-det_bbox_xyxy[0])
        h  = max(1e-3, det_bbox_xyxy[3]-det_bbox_xyxy[1])
        s  = w*h; r = w/h
        z = np.array([cx, cy, s, r], dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    def to_xyxy(self) -> List[float]:
        cx, cy, s, r = self.x[0], self.x[1], max(1e-3,self.x[2]), max(1e-3,self.x[3])
        w = math.sqrt(s*r); h = s/max(w,1e-3)
        return [float(cx-w/2), float(cy-h/2), float(cx+w/2), float(cy+h/2)]

# ========== 4) Aparycja (HSV histogram + cosine) ==========
def hsv_hist(bgr: np.ndarray, bins=(8,8,4)) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256]).flatten()
    hist /= (np.linalg.norm(hist) + 1e-8)
    return hist

def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8))

# ========== 5) Tracker ==========
@dataclass
class Track:
    track_id: int
    label: int
    score: float
    kf: KalmanBox
    bbox: List[float]            # ostatni bbox (EMA)
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    app: Optional[np.ndarray] = None  # appearance

def try_linear_assignment(cost: np.ndarray) -> np.ndarray:
    # Zwraca Nx2 (rows, cols); jeśli brak scipy -> greedy fallback
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return np.array(list(zip(r, c)), dtype=int)
    except Exception:
        # Greedy fallback
        C = cost.copy()
        matches = []
        used_r, used_c = set(), set()
        while True:
            i, j = np.unravel_index(np.argmin(C), C.shape)
            if np.isinf(C[i,j]): break
            if i in used_r or j in used_c:
                C[i,j] = np.inf
                continue
            matches.append((i,j))
            used_r.add(i); used_c.add(j)
            C[i,:] = np.inf; C[:,j] = np.inf
        return np.array(matches, dtype=int)

class KalmanHungarianTracker:
    def __init__(self,
                 iou_gate=Cfg.iou_gate,
                 mhal_gate=Cfg.mhal_gate,
                 alpha_cost=Cfg.alpha_cost,
                 max_age=Cfg.max_age,
                 min_hits=Cfg.min_hits,
                 ema_alpha=Cfg.ema_alpha,
                 use_appearance=Cfg.use_appearance,
                 app_bins=Cfg.app_bins,
                 app_beta=Cfg.app_beta,
                 app_gate=Cfg.app_gate):
        self.iou_gate = iou_gate
        self.mhal_gate = mhal_gate
        self.alpha_cost = alpha_cost
        self.max_age = max_age
        self.min_hits = min_hits
        self.ema_alpha = ema_alpha
        self.use_app = use_appearance
        self.app_bins = app_bins
        self.app_beta = app_beta
        self.app_gate = app_gate
        self.tracks: List[Track] = []
        self.next_id = 1

    def _extract_app(self, frame: np.ndarray, bbox_xyxy: List[float]) -> Optional[np.ndarray]:
        if not self.use_app:
            return None
        x1,y1,x2,y2 = map(int, bbox_xyxy)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1: return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None
        return hsv_hist(roi, bins=self.app_bins)

    def _merge_duplicate_tracks(self, iou_thr=0.8):
        # jeśli dwa tory (ta sama klasa) bardzo się pokrywają, zostaw starszy/stabilniejszy
        keep = []
        tracks = sorted(self.tracks, key=lambda t: (t.hits, -t.time_since_update), reverse=True)
        suppressed = set()
        for i in range(len(tracks)):
            if i in suppressed: continue
            ti = tracks[i]
            for j in range(i + 1, len(tracks)):
                if j in suppressed: continue
                tj = tracks[j]
                if ti.label != tj.label: continue
                if iou_xyxy(ti.bbox, tj.bbox) >= iou_thr:
                    # zachowaj "lepszego": więcej hits -> stabilniejszy
                    suppressed.add(j)
            # nic go nie zabiło
        for k, t in enumerate(tracks):
            if k not in suppressed:
                keep.append(t)
        self.tracks = keep

    def step(self, frame: np.ndarray, dets: List[Dict]) -> List[Dict]:
        H, W = frame.shape[:2]
        # 1) Predykcja
        for t in self.tracks:
            t.bbox = clamp_bbox(t.kf.predict(), W, H)
            t.age += 1
            t.time_since_update += 1

        # 2) Dopasowanie przez koszt łączony
        matches, unmatched_tracks, unmatched_dets = self._match(frame, dets)
        if unmatched_dets:
            matched_det_idxs = {dj for _, dj in matches}
            matched_dets = [dets[i] for i in matched_det_idxs]
            new_unmatched = []
            for dj in unmatched_dets:
                d = dets[dj]
                dup = False
                for md in matched_dets:
                    # jeśli bardzo podobny bbox tej samej klasy – potraktuj jako duplikat i pomiń
                    if d["label"] == md["label"] and iou_xyxy(d["bbox"], md["bbox"]) >= 0.65:
                        dup = True
                        break
                if not dup:
                    new_unmatched.append(dj)
            unmatched_dets = new_unmatched

        # 3) Update dopasowanych
        for ti, dj in matches:
            det = dets[dj]
            self.tracks[ti].kf.update(det["bbox"])
            meas_bbox = clamp_bbox(self.tracks[ti].kf.to_xyxy(), W, H)

            # EMA smoothing
            if self.tracks[ti].bbox is None:
                self.tracks[ti].bbox = meas_bbox
            else:
                a = self.ema_alpha
                self.tracks[ti].bbox = [a*self.tracks[ti].bbox[k] + (1-a)*meas_bbox[k] for k in range(4)]

            self.tracks[ti].score = det["score"]
            self.tracks[ti].hits += 1
            self.tracks[ti].time_since_update = 0

            # Aparycja (aktualizacja)
            app = self._extract_app(frame, self.tracks[ti].bbox)
            if app is not None:
                if self.tracks[ti].app is None:
                    self.tracks[ti].app = app
                else:
                    self.tracks[ti].app = 0.8*self.tracks[ti].app + 0.2*app

        # 4) Nowe tory z niedopasowanych detekcji
        for dj in unmatched_dets:
            d = dets[dj]
            kf = KalmanBox(d["bbox"])
            app = self._extract_app(frame, d["bbox"])
            self.tracks.append(Track(track_id=self.next_id, label=d["label"], score=d["score"],
                                     kf=kf, bbox=clamp_bbox(kf.to_xyxy(), W, H), app=app))
            self.next_id += 1

        # 5) Usuwanie starych
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 6) Wynik
        out = []
        for t in self.tracks:
            out.append({
                "track_id": t.track_id,
                "label": t.label,
                "bbox": t.bbox,
                "score": t.score,
                "confirmed": t.hits >= self.min_hits
            })
        self._merge_duplicate_tracks(iou_thr=0.80)
        return out

    def _match(self, frame: np.ndarray, dets: List[Dict]):
        if not self.tracks or not dets:
            return [], list(range(len(self.tracks))), list(range(len(dets)))

        T, D = len(self.tracks), len(dets)
        INF = 1e9
        cost = np.full((T, D), INF, dtype=float)

        for ti, t in enumerate(self.tracks):
            for dj, d in enumerate(dets):
                if d["label"] != t.label:        # bramkowanie po klasie
                    continue
                # Bramkowanie Mahalanobisem
                mhal = t.kf.gating_distance(d["bbox"])
                if mhal > self.mhal_gate:
                    continue
                # Bramkowanie IoU
                iou = iou_xyxy(t.bbox, d["bbox"])
                if iou < self.iou_gate:
                    continue
                c = self.alpha_cost * mhal + (1.0 - self.alpha_cost) * (1.0 - iou)

                # Aparycja (koszt kosinusowy)
                if self.use_app:
                    # Ekstrakcja app tylko gdy tor ją ma (by nie liczyć bez potrzeby)
                    if t.app is not None:
                        # Szybkie wycięcie ROI z detekcji
                        x1,y1,x2,y2 = map(int, d["bbox"])
                        x1 = max(0, min(x1, frame.shape[1]-1))
                        x2 = max(0, min(x2, frame.shape[1]-1))
                        y1 = max(0, min(y1, frame.shape[0]-1))
                        y2 = max(0, min(y2, frame.shape[0]-1))
                        if x2 > x1 and y2 > y1:
                            roi = frame[y1:y2, x1:x2]
                            app_det = hsv_hist(roi, bins=self.app_bins)
                            cd = cosine_dist(t.app, app_det)
                            if cd > self.app_gate:
                                continue
                            c += self.app_beta * cd

                cost[ti, dj] = c

        matches_idx = try_linear_assignment(cost)
        used_t, used_d, matches = set(), set(), []
        for ti, dj in matches_idx:
            if cost[ti, dj] >= INF:
                continue
            used_t.add(int(ti)); used_d.add(int(dj))
            matches.append((int(ti), int(dj)))

        unmatched_tracks = [i for i in range(T) if i not in used_t]
        unmatched_dets = [j for j in range(D) if j not in used_d]
        return matches, unmatched_tracks, unmatched_dets

def demo(video_source: Optional[str | int] = 0, show_names=True):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Nie mogę otworzyć wideo/kamery")

    detector = RTDetrWrapper(weights=Cfg.weights, score_thresh=Cfg.conf_det,
                             imgsz=Cfg.imgsz, iou_nms=Cfg.iou_nms, max_det=Cfg.max_det)
    tracker = KalmanHungarianTracker()

    t0 = time.time()
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        dets = detector(frame)
        dets = nms_per_class(dets, iou_thr=0.5)
        tracks = tracker.step(frame, dets)

        for tr in tracks:
            x1, y1, x2, y2 = map(int, tr["bbox"])
            clr = (0, 255, 0) if tr["confirmed"] else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
            label = str(tr["label"])
            text = f'ID {tr["track_id"]} cls {label}'
            cv2.putText(frame, text, (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)

        frames += 1
        if frames % 20 == 0:
            fps = frames / (time.time() - t0 + 1e-9)
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)

        cv2.imshow("RT-DETR + Kalman + Hungarian", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np
    import cv2 as cv

    cap = cv.VideoCapture(0)

    ret, frame = cap.read()
    bbox = cv.selectROI('select', frame, False)

    x, y, w, h = bbox

    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)),
                      np.array((180., 255., 255.)))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

    while (1):
        ret, frame = cap.read()

        if ret == True:
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            ret, track_window = cv.meanShift(dst, bbox, term_crit)

            x, y, w, h = track_window
            img2 = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv.imshow('gfg', img2)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()
    # demo(0)  # kamera