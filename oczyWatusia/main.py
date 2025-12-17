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

os.environ.setdefault("HF_HOME", "/media/jetson/sd/hg")

GPU_ENABLED = True

if GPU_ENABLED:
    from src.img_classifiers.image_classifier import getClassifiers, getClothesClassifiers

from src import calc_brightness, calc_obj_angle, suggest_mode
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
                if isinstance(el, dict):
                    res += "\t"*(indent+1) + pretty_print_dict(el, indent + 1) + ",\n"
                else:
                    res += "\t"*(indent+1) + str(el) + ",\n"
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
            use_net_stream: bool = True
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
        if GPU_ENABLED:
            self.emotion_classifier, self.gender_classifier, self.age_classifier = getClassifiers()
            self.clothes_type_clf, self.clothes_pattern_clf, self.color_clf = getClothesClassifiers()
        self.person_cache = {} # track_id -> {last_frame: int, gender: str, age: str, emotion: str}

        if cap is None:
            if use_net_stream:
                # GStreamer pipeline for network stream
                gst_pipeline = (
                    "udpsrc port=5000 ! "
                    "application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96 ! "
                    "rtph264depay ! h264parse ! nvv4l2decoder ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! "
                    "appsink drop=1"
                )
                print(f"Opening network stream with pipeline:\n{gst_pipeline}")
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            else:
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
        clothes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/clothes.pt')
        guns_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/guns.pt')
        self.clothes_detector = YOLO(clothes_path)  
        self.guns_detector = YOLO(guns_path)
        
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
        fov_deg: int = 102,
        consolidate_with_lidar: bool = False,
    ):
        lidar_path = "../lidar/data/lidar.json"
        lidar_tracks_data = []

        def get_lidar_tracks():
            """Reads the last line of lidar.json"""
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
                    tracks = data.get("tracks", [])
                    # print(f"DEBUG: Lidar tracks read: {len(tracks)}")
                    return tracks
            except Exception as e:
                print(f"Lidar read error: {e}")
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
        weapon_detection_enabled = False
        self.last_detected_guns = []

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
                                # print(f"DEBUG_MATCH: P_ID={p_idx} P_ANG={p_ang:.2f} L_ID={lt.get('id')} L_POS={l_pos} L_ANG={l_ang:.2f} DIFF={diff:.2f}")
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
                        add_info = []

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
                                current_h, current_w = person_crop.shape[:2]
                                
                                # Check cache
                                cache_entry = self.person_cache.get(track_id, {"last_frame": -9999})
                                should_update = (self.frame_idx - cache_entry["last_frame"]) > 100

                                if should_update:
                                    # --- UPDATE CACHE (Clothes + Classifiers) ---
                                    new_cache = {
                                        "last_frame": self.frame_idx,
                                        "clothes": [],
                                        "emotion": cache_entry.get("emotion"),
                                        "gender": cache_entry.get("gender"),
                                        "age": cache_entry.get("age"),
                                        "lidar_data": cache_entry.get("lidar_data")
                                    }

                                    # 1. CLOTHES
                                    results_clothes = self.clothes_detector.predict(person_crop, verbose=False)
                                    clothes_data = []
                                    for rc in results_clothes:
                                        c_boxes = rc.boxes.xyxy.cpu().numpy()
                                        c_clss  = rc.boxes.cls.cpu().numpy()
                                        c_conf  = rc.boxes.conf.cpu().numpy()
                                        c_names = rc.names
                                        for cb, cc, ccnf in zip(c_boxes, c_clss, c_conf):
                                            # Normalize coordinates (0-1) relative to crop
                                            cx1, cy1, cx2, cy2 = cb
                                            norm_box = [float(cx1)/current_w, float(cy1)/current_h, float(cx2)/current_w, float(cy2)/current_h]
                                            
                                            details = {}
                                            label_name = c_names[int(cc)]
                                            
                                            
                                            # Sub-crop for the specific item
                                            # Coordinates are relative to person_crop
                                            icx1, icy1, icx2, icy2 = int(cx1), int(cy1), int(cx2), int(cy2)
                                            # Clamp
                                            icx1 = max(0, icx1); icy1 = max(0, icy1)
                                            icx2 = min(current_w, icx2); icy2 = min(current_h, icy2)
                                            
                                            if icx2 > icx1 and icy2 > icy1:
                                                item_crop = person_crop[icy1:icy2, icx1:icx2]
                                                try:
                                                    item_pil = Image.fromarray(cv2.cvtColor(item_crop, cv2.COLOR_BGR2RGB))
                                                    
                                                    # Color for all relevant classes
                                                    if label_name in ["clothing", "shoe", "bag"] and GPU_ENABLED:
                                                        # color_clf returns tuple (R, G, B) or similar
                                                        # Assuming simple tuple or string logic. 
                                                        # User said "zwraca krotkę w RGB". We should probably format it.
                                                        
                                                        dom_color = self.color_clf(np.asarray(item_pil))
                                                        details["color"] = [int(x) for x in dom_color]
                                                    
                                                    # Details for "clothing"
                                                    # if label_name == "clothing" and GPU_ENABLED:
                                                    #     res_type = self.clothes_type_clf.process(item_pil)
                                                    #     details["type"] = max(res_type, key=res_type.get)
                                                        
                                                    #     res_pat = self.clothes_pattern_clf.process(item_pil)
                                                    #     details["pattern"] = max(res_pat, key=res_pat.get)
                                                        
                                                except Exception as e:
                                                    print(f"Clothes detail error: {e}")

                                            clothes_data.append({
                                                "box_norm": norm_box,
                                                "label": label_name,
                                                "class_num": int(cc),
                                                "conf": float(ccnf),
                                                "details": details
                                            })
                                    new_cache["clothes"] = clothes_data

                                    # 2. CLASSIFIERS
                                    try:
                                        pil_img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))

                                        if GPU_ENABLED and False:
                                            res_emo = self.emotion_classifier.process(pil_img)
                                            new_cache["emotion"] = max(res_emo, key=res_emo.get)

                                            res_gen = self.gender_classifier.process(pil_img)
                                            new_cache["gender"] = max(res_gen, key=res_gen.get)

                                            res_age = self.age_classifier.process(pil_img)
                                            new_cache["age"] = max(res_age, key=res_age.get)
                                    except Exception as e:
                                        print(f"Classifier error: {e}")
                                    
                                    self.person_cache[track_id] = new_cache
                                    cache_entry = new_cache

                                # --- DRAW FROM CACHE ---
                                # Draw Clothes
                                if "clothes" in cache_entry:
                                    for item in cache_entry["clothes"]:
                                        nx1, ny1, nx2, ny2 = item["box_norm"]
                                        label_txt = f"{item['label']} {item['conf']:.2f}"
                                        
                                        # Denormalize to current frame global coordinates
                                        gx1 = int(b_x1 + nx1 * current_w)
                                        gy1 = int(b_y1 + ny1 * current_h)
                                        gx2 = int(b_x1 + nx2 * current_w)
                                        gy2 = int(b_y1 + ny2 * current_h)

                                        color = (0, 0, 255)
                                        cv2.rectangle(frame_bgr, (gx1, gy1), (gx2, gy2), color, 2)
                                        cv2.putText(frame_bgr, label_txt, (gx1, gy1 - 5),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                                # Attributes info
                                add_info = []
                                if "gender" in cache_entry and cache_entry["gender"]: 
                                    add_info.append({"gender": cache_entry["gender"]})
                                if "age" in cache_entry and cache_entry["age"]: 
                                    add_info.append({"age": cache_entry["age"]})
                                if "emotion" in cache_entry and cache_entry["emotion"]: 
                                    add_info.append({"emotion": cache_entry["emotion"]})
                                if "clothes" in cache_entry and cache_entry["clothes"]:
                                    add_info.append({"clothes": cache_entry["clothes"]})

                        
                        # --- LIDAR INFO ---
                        lidar_info = {}
                        l_id = None
                        l_dist = None
                        
                        # 1. Try to get new match
                        if i in lidar_matches:
                            lm = lidar_matches[i]
                            l_id = lm.get("id", "?")
                            l_dist = lm.get("last_position", [0, 0])[1] # y is forward
                            
                            # Save to cache
                            if track_id not in self.person_cache:
                                self.person_cache[track_id] = {"last_frame": -9999}
                            self.person_cache[track_id]["lidar_data"] = {"id": l_id, "dist": l_dist}
                        
                        # 2. If no new match, check cache
                        elif track_id in self.person_cache and self.person_cache[track_id].get("lidar_data"):
                            cached_l = self.person_cache[track_id]["lidar_data"]
                            l_id = cached_l["id"]
                            l_dist = cached_l["dist"]

                        if l_id is not None:
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

                # --- WEAPON DETECTION ---
                if weapon_detection_enabled and (self.frame_idx % 10 == 0):
                    with torch.inference_mode():
                        res_guns = self.guns_detector(frame_bgr, verbose=False)
                    
                    self.last_detected_guns = []
                    for rg in res_guns:
                        for box in rg.boxes:
                            gx1, gy1, gx2, gy2 = box.xyxy[0].cpu().numpy()
                            g_conf = float(box.conf[0].cpu().numpy())
                            g_cls_id = int(box.cls[0].cpu().numpy())
                            g_label = rg.names[g_cls_id]
                            self.last_detected_guns.append({
                                "box": [gx1, gy1, gx2, gy2],
                                "conf": g_conf,
                                "label": g_label
                            })
                
                # Draw and add to detections (if enabled and present)
                if weapon_detection_enabled:
                    for gun in self.last_detected_guns:
                        gx1, gy1, gx2, gy2 = gun["box"]
                        g_conf = gun["conf"]
                        g_label = gun["label"]

                        # Draw BLACK box (0, 0, 0)
                        if show_window:
                            cv2.rectangle(frame_bgr, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 0, 0), 2)
                            cv2.putText(frame_bgr, f"{g_label} {g_conf:.2f}", (int(gx1), int(gy1)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        detections["objects"].append({
                            "id": -1,
                            "type": g_label,
                            "left": float(gx1), "top": float(gy1),
                            "width": float(gx2-gx1), "height": float(gy2-gy1),
                            "isPerson": False,
                            "angle": 0.0,
                            "additionalInfo": [{"type": "weapon"}],
                            "lidar": {}
                        })
                            
                ema_fps = self.calc_fps() if show_fps else 0

                height = frame_bgr.shape[0]

                cv2.rectangle(frame_bgr, (0, height), (200,height-95), (0, 0, 0), -1) if verbose_window else None
                cv2.putText(frame_bgr, f"FPS: {ema_fps:0,.2f}", (0,height-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,255), 2) if verbose_window else None
                cv2.putText(frame_bgr, f"Light: {detections['brightness']:0,.2f}",
                            (0, height-30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255), 2) if verbose_window else None
                cv2.putText(frame_bgr, f"Mode: {detections['suggested_mode']}",
                            (0, height),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255), 2) if verbose_window else None

                self.save_to_json("camera.jsonl", detections) if self.save_to_json is not None else None
                # print(f"Detections: ", pretty_print_dict(detections), f"FPS: {ema_fps:.1f}") if verbose \
                    # else None

                cv2.imshow(self.window_name, frame_bgr) if show_window else None

                self.video_recorder.write(frame_bgr) if save_video else None
                self.frame_idx += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord(ESCAPE_BUTTON):
                    break
                elif key == ord('w'):
                    weapon_detection_enabled = not weapon_detection_enabled
                    print(f"Weapon Detection: {'ON' if weapon_detection_enabled else 'OFF'}")

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
