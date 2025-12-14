import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import json
import math
from PIL import Image
from ultralytics import YOLO

from oczyWatusia.src.img_classifiers.image_classifier import getClassifiers

from oczyWatusia.src import calc_brightness, calc_obj_angle, suggest_mode
from dotenv import load_dotenv
from torch.amp import autocast
load_dotenv()

# Paleta (RGB w [0,1]); do OpenCV zamienimy na BGR w [0,255]
COLORS = np.array([
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
] * 100, dtype=np.float32)

COLORS_BGR = (COLORS[:, ::-1] * 255.0).astype(np.uint8)
ESCAPE_BUTTON = "q"


def pretty_print_dict(d, indent=1):
    res = "\n"
    for k, v in d.items():
        res += "\t"*indent + str(k)
        if isinstance(v, list):
            res += "[\n"
            for el in v:
                res += "\t"*(indent+1) + pretty_print_dict(el, indent + 1) + ",\n"
            res += "]"
        elif isinstance(v, dict):
            res += "\n" + pretty_print_dict(v, indent+1)
        else:
            res += "\t"*(indent+1) + str(v) + "\n"
    return res

class CVAgent:
    def __init__(
            self,
            weights_path: str = "yolo12s.pt",
            imgsz: int = 640,
            source: int | str = 0,
            cap=None,
            json_save_func=None,
        ):
        self.save_to_json = json_save_func
        self.imgsz = imgsz
        self.track_history = defaultdict(lambda: [])

        cv2.setUseOptimized(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        
        # Load Classifiers
        self.emotion_classifier, self.gender_classifier, self.age_classifier = getClassifiers()
        self.person_cache = {} # track_id -> {last_frame: int, gender: str, age: str, emotion: str}

        if cap is None:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print("Nie mogę otworzyć kamery")
                return
        else:
            self.cap = cap
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_recorder = None


        self.fps_params = {
            "ema_alpha": 0.1,
            "last_stat_t": time.time(),
            "t_prev": 0,
            "show_fps_every": 0.5
        }
        self.frame_idx = 0

        self.mil_vehicles_details = {}
        self.clothes_details = {}

        self.detector = YOLO(weights_path)
        self.class_names = self.detector.names
        
        # Inicjalizacja modelu do ubrań
        clothes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/cv/best.pt')
        self.clothes_detector = YOLO(clothes_path)
        
        self.window_name = f"YOLOv12 – naciśnij '{ESCAPE_BUTTON}' aby wyjść"

    def init_recorder(self, out_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_cap = self.cap.get(cv2.CAP_PROP_FPS)
        fps_output = float(fps_cap) if fps_cap and fps_cap > 1.0 else 20.0
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_recorder = cv2.VideoWriter(out_path, fourcc, fps_output, (width, height))
        if self.video_recorder is None:
            return False
        else:
            return True

    def init_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def actualize_tracks(self, frame_bgr, track_id, point: tuple[int, int]):
        x, y = point
        track = self.track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 30:
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(frame_bgr, [points], False, color=(230, 230, 230), thickness=10)

    def calc_fps(self):
        ema_alpha, last_stat_t, t_prev, show_fps_every = self.fps_params.values()
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - t_prev))
        self.fps_params["t_prev"] = now
        ema_fps = 0.0
        ema_alpha = 0.1
        ema_fps = (1 - ema_alpha) * ema_fps + ema_alpha * inst_fps if ema_fps > 0 else inst_fps

        # Overlay
        if (now - last_stat_t) >= show_fps_every:
            self.fps_params["last_stat_t"] = now
        return ema_fps

    def warm_up_model(self):
        _ret, _warm = self.cap.read()
        if _ret:
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with autocast(dtype=torch.float32, device_type=self.device.type):
                        _ = self.detector(_warm)
                else:
                    _ = self.detector(_warm)

    def detect_objects(self, frame_bgr, imgsz: int = 640, run_detection=True):
        if run_detection:
            iou = 0.7
            conf = 0.3
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with autocast(dtype=torch.float32, device_type=self.device.type):
                        detections = self.detector.track(frame_bgr, persist=True,
                                                         device=self.device, verbose=False,
                                                  imgsz=imgsz, iou=iou, conf=conf)
                else:
                    detections = self.detector.track(frame_bgr, persist=True,
                                                     device=self.device, verbose=False,
                                              imgsz=imgsz, iou=iou, conf=conf)
        return detections[0]

    def run(
        self,
        save_video: bool = False,
        out_path: str = "output.mp4",
        show_window=True,
        det_stride: int = 1,
        show_fps: bool = True,
        verbose: bool = True,
        verbose_window: bool = True,
        fov_deg: int = 60,
        consolidate_with_lidar: bool = False,
    ):
        lidar_path = "lidar.jsonl"
        lidar_tracks_data = []

        def get_lidar_tracks():
            """Reads the last line of lidar.jsonl"""
            try:
                if not os.path.exists(lidar_path):
                    return []
                # Efficiently read last line - for now just readlines (simple) or Seek
                # Since we don't have a giant file helper, standard read is okay for small buffer or we assume append
                with open(lidar_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        return []
                    last_line = lines[-1].strip()
                    if not last_line:
                        return []
                    data = json.loads(last_line)
                    return data.get("tracks", [])
            except Exception as e:
                # print(f"Lidar read error: {e}")
                return []
        
        def compute_lidar_angle_local(lx, ly):
            # angle in degrees
            return math.degrees(math.atan2(lx, ly))

        save_video = self.init_recorder(out_path) if save_video else None
        if show_window is False or show_window is None:
            verbose_window = False

        self.init_window() if show_window else None

        detections = {
            "objects": [],
            "countOfPeople": 0,
            "countOfObjects": 0,
            "suggested_mode": '',
            "brightness": 0.0,
        }
        self.frame_idx = 0
        mode = 'light'

        self.warm_up_model()

        try:
            self.fps_params["t_prev"] = time.time()

            while True:
                ret, frame_bgr = self.cap.read()
                if not ret:
                    print("Koniec strumienia")
                    break

                detections["countOfPeople"] = 0
                detections["countOfObjects"] = 0
                detections["objects"] = []

                run_detection = (self.frame_idx % det_stride == 0)

                dets = self.detect_objects(frame_bgr, run_detection=run_detection)

                detections["brightness"] = calc_brightness(frame_bgr)
                detections["suggested_mode"] = suggest_mode(detections["brightness"], mode)

                if dets.boxes and dets.boxes.is_track:
                    boxes = dets.boxes.xywh.cpu()
                    boxes_xyxy = dets.boxes.xyxy.cpu()  # Potrzebne do wycinania
                    track_ids = dets.boxes.id.int().cpu().tolist()
                    labels = dets.boxes.cls.int().cpu().tolist()
                    
                    # --- LIDAR SYNC ---
                    lidar_matches = {} # map track_id -> lidar_track_dict
                    if consolidate_with_lidar:
                        l_tracks = get_lidar_tracks()
                        # Simple greedy matching
                        used_lidar_indices = set()
                        
                        # Precompute camera angles for all people
                        people_indices = [idx for idx, lbl in enumerate(labels) if lbl == 0]
                        
                        person_angles = []
                        for idx in people_indices:
                             _x, _y, _w, _h = boxes[idx]
                             _ang = calc_obj_angle((_x, _y), (_x + _w, _y + _h), self.imgsz, fov_deg=fov_deg)
                             person_angles.append((idx, _ang))
                             
                        # Try to match each person to closest lidar track
                        for p_idx, p_ang in person_angles:
                            best_lidar_idx = -1
                            min_diff = 1000.0
                            
                            for l_idx, lt in enumerate(l_tracks):
                                if l_idx in used_lidar_indices:
                                    continue
                                l_pos = lt.get("last_position", [0, 0])
                                l_ang = compute_lidar_angle_local(l_pos[0], l_pos[1])
                                
                                diff = abs(p_ang - l_ang)
                                if diff < min_diff and diff < 20.0: # Tolerance
                                    min_diff = diff
                                    best_lidar_idx = l_idx
                            
                            if best_lidar_idx != -1:
                                used_lidar_indices.add(best_lidar_idx)
                                # Map camera track_id (if exists) or just index to lidar data
                                # Here we map for the loop below
                                lidar_matches[p_idx] = l_tracks[best_lidar_idx]

                    frame_bgr = dets.plot() if show_window else frame_bgr

                    for i, (box, track_id, label) in enumerate(zip(boxes, track_ids, labels)):
                        x, y, w, h = box
                        angle = calc_obj_angle((x, y), (x + w, y + h), self.imgsz, fov_deg=fov_deg)

                        self.actualize_tracks(frame_bgr, track_id, (x, y)) if verbose_window else None

                        # --- CLOTHES DETECTION ON PERSON ---
                        if label == 0:  # Person
                            b_x1, b_y1, b_x2, b_y2 = map(int, boxes_xyxy[i].tolist())
                            # Clamp
                            H_frame, W_frame = frame_bgr.shape[:2]
                            b_x1 = max(0, b_x1); b_y1 = max(0, b_y1)
                            b_x2 = min(W_frame, b_x2); b_y2 = min(H_frame, b_y2)

                            if b_x2 > b_x1 and b_y2 > b_y1:
                                person_crop = frame_bgr[b_y1:b_y2, b_x1:b_x2]
                                
                                # --- CLOTHES DETECTION ---
                                # Inference
                                results_clothes = self.clothes_detector.predict(person_crop, verbose=False)
                                for rc in results_clothes:
                                    c_boxes = rc.boxes.xyxy.cpu().numpy()
                                    c_clss  = rc.boxes.cls.cpu().numpy()
                                    c_conf  = rc.boxes.conf.cpu().numpy()
                                    c_names = rc.names

                                    for cb, cc, ccnf in zip(c_boxes, c_clss, c_conf):
                                        cx1, cy1, cx2, cy2 = cb
                                        # Transform to global
                                        gx1 = int(b_x1 + cx1)
                                        gy1 = int(b_y1 + cy1)
                                        gx2 = int(b_x1 + cx2)
                                        gy2 = int(b_y1 + cy2)

                                        # Draw
                                        color = (0, 0, 255) # Red for clothes
                                        cv2.rectangle(frame_bgr, (gx1, gy1), (gx2, gy2), color, 2)
                                        label_txt = f"{c_names[int(cc)]} {ccnf:.2f}"
                                        cv2.putText(frame_bgr, label_txt, (gx1, gy1 - 5),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                                # --- CLASSIFIERS (Age, Gender, Emotion) ---
                                # Check cache
                                cache_entry = self.person_cache.get(track_id, {"last_frame": -9999})
                                if (self.frame_idx - cache_entry["last_frame"]) > 100:
                                     # Convert to PIL
                                    try:
                                        # CV2 is BGR, PIL needs RGB
                                        pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                                        
                                        # Run Classifiers
                                        # process returns dict {label: prob}. We take max key.
                                        res_emo = self.emotion_classifier.process(pil_img)
                                        emo_label = max(res_emo, key=res_emo.get)
                                        
                                        res_gen = self.gender_classifier.process(pil_img)
                                        gen_label = max(res_gen, key=res_gen.get)
                                        
                                        res_age = self.age_classifier.process(pil_img)
                                        age_label = max(res_age, key=res_age.get)
                                        
                                        cache_entry = {
                                            "last_frame": self.frame_idx,
                                            "emotion": emo_label,
                                            "gender": gen_label,
                                            "age": age_label
                                        }
                                        self.person_cache[track_id] = cache_entry
                                    except Exception as e:
                                        print(f"Classifier error: {e}")

                        # Use cached info if available, even if we didn't update valid this frame
                        p_info = self.person_cache.get(track_id, {})
                        
                        # Pack additional info
                        add_info = []
                        if "gender" in p_info: add_info.append({"gender": p_info["gender"]})
                        if "age" in p_info: add_info.append({"age": p_info["age"]})
                        if "emotion" in p_info: add_info.append({"emotion": p_info["emotion"]})

                        
                        # --- LIDAR INFO ---
                        lidar_info = {}
                        if i in lidar_matches:
                            lm = lidar_matches[i]
                            l_id = lm.get("id", "?")
                            l_dist = lm.get("last_position", [0, 0])[1] # y is forward
                            lidar_info = {"lidar_id": l_id, "distance": l_dist}
                            
                            # Draw on frame
                            txt_lidar = f"LIDAR ID: {l_id} Dist: {l_dist:.2f}m"
                            cv2.putText(frame_bgr, txt_lidar, (int(x), int(y) - 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


                        detections["objects"].append({
                            "id": track_id,
                            "type": self.class_names[label],
                            "left": x.item(),
                            "top": y.item(),
                            "width": w.item(),
                            "height": h.item(),
                            "isPerson": True if label == 0 else False,
                            "angle": angle,
                            "additionalInfo": add_info,
                            "lidar": lidar_info
                        })
                        detections["countOfObjects"] += 1
                        detections["countOfPeople"] += (1 if label == 0 else 0)

                ema_fps = self.calc_fps() if show_fps else 0

                height = frame_bgr.shape[0]

                cv2.rectangle(frame_bgr, (0, height), (200,height-95), (0, 0, 0), -1) if verbose_window else None
                cv2.putText(frame_bgr, f"FPS: {ema_fps:0,.2f}", (0,height-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,255), 2) if verbose_window else None
                cv2.putText(frame_bgr, f"Light: {detections["brightness"]:0,.2f}",
                            (0, height-30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255), 2) if verbose_window else None
                cv2.putText(frame_bgr, f"Mode: {detections["suggested_mode"]}",
                            (0, height),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255), 2) if verbose_window else None

                self.save_to_json("camera.jsonl", detections) if self.save_to_json is not None else None
                print(f"Detections: ", pretty_print_dict(detections), f"FPS: {ema_fps:.1f}") if verbose \
                    else None

                cv2.imshow(self.window_name, frame_bgr) if show_window else None

                self.video_recorder.write(frame_bgr) if save_video else None
                self.frame_idx += 1

                if cv2.waitKey(1) & 0xFF == ord(ESCAPE_BUTTON):
                    break

        except KeyboardInterrupt:
            print("Przerwano przez użytkownika.")
        finally:
            self.cap.release()
            if self.video_recorder is not None:
                self.video_recorder.release()
            if show_window:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    agent = CVAgent()
    agent.run(save_video=True, show_window=True, consolidate_with_lidar=True)
