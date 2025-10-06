# -*- coding: utf-8 -*-
# Zewnętrzne API pakietu img_classifiers

from .color_classifier import dominant_colors as detect_color

# Niektóre paczki mogą nie mieć gender_classifier albo odpowiedniej funkcji.
# Zapewniamy bezpieczny fallback, żeby kamera się nie wywalała.
try:
    from .gender_classifier import detect_gender
except Exception:
    def detect_gender(*args, **kwargs):
        return {"gender": "unknown", "conf": 0.0}
