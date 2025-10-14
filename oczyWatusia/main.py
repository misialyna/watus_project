import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# from src.img_classifiers import detect_color, detect_gender
from src import CVWrapper, Tracker, calc_brightness, calc_obj_angle, suggest_mode
from src import nms_per_class
from dotenv import load_dotenv
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

def detectFromCamera(
    score_thresh: float = 0.15,
    iou_thresh: float = 0.55,
    weights: str = "yolo12s.pt",
    imgsz: int = 1280,
    save_video: bool = False,
    out_path: str = "output.mp4",
    cam_index: int = 0,
    det_stride: int = 2,          # <— detektor co N klatek
    show_fps_every: float = 0.5,

):

    cv2.setUseOptimized(True)
    # Urządzenie + drobne usprawnienia wydajnościowe
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Nie mogę otworzyć kamery")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        fps_output = float(fps_cap) if fps_cap and fps_cap > 1.0 else 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps_output, (width, height))

    window_name = "YOLOv12 – naciśnij 'q' aby wyjść"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    mode = 'light'
    # detector = CVWrapper(weights=weights, score_thresh=score_thresh, imgsz=imgsz)
    tracker = Tracker()
    detector = YOLO("yolo12s.pt")

    detections = {
        "objects": [],
        "countOfPeople": 0,
        "countOfObjects": 0,
    }
    class_names = detector.names  # lokalny uchwyt (szybciej)

    ema_fps = 0.0
    ema_alpha = 0.1
    last_stat_t = time.time()
    frame_idx = 0

    # Warm-up (zmniejsza "pierwszo-klatkowego" laga)
    _ret, _warm = cap.read()
    if _ret:
        with torch.inference_mode():
            if device.type == "cuda":
                from torch.amp import autocast
                with autocast(dtype=torch.float32, device_type=device.type):
                    _ = detector(_warm)
            else:
                _ = detector(_warm)

    try:
        t_prev = time.time()

        track_history = defaultdict(lambda: [])
        # Pętla
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Koniec strumienia")
                break

            detections["countOfPeople"] = 0
            detections["countOfObjects"] = 0
            detections["objects"] = []
            H, W = frame_bgr.shape[:2]

            run_detection = (frame_idx % det_stride == 0)

            with torch.inference_mode():
                if device.type == "cuda":
                    from torch.amp import autocast
                    with autocast(dtype=torch.float32, device_type=device.type):
                        dets = detector.track(frame_bgr, persist=True, device=device, verbose=False, imgsz=imgsz)[0]
                else:
                        dets = detector.track(frame_bgr, persist=True, device=device, verbose=False, imgsz=imgsz)[0]
            #         dets = nms_per_class(dets)
            #     tracks = tracker.step(frame_bgr, dets)
            # else:
            #     dets = nms_per_class(dets)
            #     try:
            #         tracks = tracker.step(frame_bgr, dets)
            #     except AttributeError:
            #         tracks = tracker.step(frame_bgr, [])

            if dets.boxes and dets.boxes.is_track:
                boxes = dets.boxes.xywh.cpu()
                track_ids = dets.boxes.id.int().cpu().tolist()
                labels = dets.boxes.cls.int().cpu().tolist()
                frame_bgr = dets.plot()

                # Szybka jasność i tryb
                brightness = calc_brightness(frame_bgr)
                mode = suggest_mode(brightness, mode)

                for box, track_id, label in zip(boxes, track_ids, labels):
                    x, y, w, h = box
                    angle = calc_obj_angle((x, y), (x + w, y + h), (W, H), 60)
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame_bgr, [points], False, color=(230, 230, 230), thickness=10)

                    detections["objects"].append({
                        "id": track_id,
                        "type": class_names[label],
                        "left": x,
                        "top": y,
                        "isPerson": True if label == 0 else False,
                        "angle": angle,
                        "additionalInfo": []
                    })
                    detections["countOfObjects"] += 1
                    detections["countOfPeople"] += (1 if label == 0 else 0)
                    print(detections)

            # kolor dla danego toru (modulo długość palety)
            # for i, tr in enumerate(tracks):
            #     x1, y1, x2, y2 = map(int, tr["bbox"])
            #     if x1 == x2 or y1 == y2 or not tr.get("confirmed", False):
            #         continue
            #
            #     # Uwaga: unikamy PIL i kosztownych kopii
            #     # crop = frame_bgr[y1:y2, x1:x2]  # jeśli kiedyś potrzebny
            #
            #     angle = calc_obj_angle((x1, y1), (x2, y2), (W, H), 60)
            #     color = tuple(int(c) for c in COLORS_BGR[i % len(COLORS_BGR)])
            #
            #     cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            #
            #     track_id = tr["track_id"]
            #     label_idx = int(tr["label"])
            #     obj_name = class_names[label_idx] if 0 <= label_idx < len(class_names) else str(label_idx)
            #     caption = f"ID {"track_id"} {obj_name}"
            #     cv2.putText(frame_bgr, caption, (x1, max(0, y1 - 7)),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            #
            #
            #
            #     count_objects += 1
            #     if label_idx == 0:
            #         count_people += 1


            # FPS (EMA)
            now = time.time()
            inst_fps = 1.0 / max(1e-6, (now - t_prev))
            t_prev = now
            ema_fps = (1 - ema_alpha) * ema_fps + ema_alpha * inst_fps if ema_fps > 0 else inst_fps

            # Overlay
            if (now - last_stat_t) >= show_fps_every:
                last_stat_t = now
            cv2.putText(frame_bgr, f"FPS: {ema_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)

            # Podgląd
            cv2.imshow(window_name, frame_bgr)

            # Opcjonalny zapis
            if out is not None:
                out.write(frame_bgr)

            frame_idx += 1

            # Wyjście klawiszem 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Przerwano przez użytkownika.")
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detectFromCamera(save_video=False, imgsz=640)
