
import cv2
import numpy as np


def calc_brightness(bgr_img):
    h, w = bgr_img.shape[:2]
    scale = max(h, w) / 160.0  # docelowo krótki bok ~160 px
    if scale > 1.0:
        frame_small = cv2.resize(bgr_img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    else:
        frame_small = bgr_img
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)

def suggest_mode(brightness, current):
    if brightness <= 33 and current == 'light':
        return 'dark'
    elif brightness > 66 and current == 'dark':
        return 'light'
    else:
        return current
