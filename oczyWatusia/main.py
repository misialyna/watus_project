import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

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
            cam_index: int = 0,
            cap=None,
        ):
        self.imgsz = imgsz
        self.track_history = defaultdict(lambda: [])

        cv2.setUseOptimized(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        if cap is None:
            self.cap = cv2.VideoCapture(cam_index)
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
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with autocast(dtype=torch.float32, device_type=self.device.type):
                        detections = self.detector.track(frame_bgr, persist=True, device=self.device, verbose=False,
                                                  imgsz=imgsz)
                else:
                    detections = self.detector.track(frame_bgr, persist=True, device=self.device, verbose=False,
                                              imgsz=imgsz)
        return detections[0]

    def run(
        self,
        save_video: bool = False,
        out_path: str = "output.mp4",
        show_window=True,
        det_stride: int = 1,
        show_fps: bool = True,
        verbose: bool = True,
        fov_deg: int = 60,
    ):
        save_video = self.init_recorder(out_path) if save_video else None
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

                if dets.boxes and dets.boxes.is_track:
                    boxes = dets.boxes.xywh.cpu()
                    track_ids = dets.boxes.id.int().cpu().tolist()
                    labels = dets.boxes.cls.int().cpu().tolist()

                    frame_bgr = dets.plot() if show_window else None

                    detections["brightness"] = calc_brightness(frame_bgr)
                    detections["suggested_mode"] = suggest_mode(detections["brightness"], mode)

                    for box, track_id, label in zip(boxes, track_ids, labels):
                        x, y, w, h = box
                        angle = calc_obj_angle((x, y), (x + w, y + h), self.imgsz, fov_deg=fov_deg)

                        self.actualize_tracks(frame_bgr, track_id, (x, y)) if show_window else None

                        detections["objects"].append({
                            "id": track_id,
                            "type": self.class_names[label],
                            "left": x,
                            "top": y,
                            "width": w,
                            "height": h,
                            "isPerson": True if label == 0 else False,
                            "angle": angle,
                            "additionalInfo": []
                        })
                        detections["countOfObjects"] += 1
                        detections["countOfPeople"] += (1 if label == 0 else 0)


                ema_fps = self.calc_fps() if show_fps else 0

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
    agent.run(save_video=False, show_window=True)
