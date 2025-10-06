from transformers import pipeline
from functools import lru_cache

import torch

@lru_cache(maxsize=1)
def _get_pipe(model_id):
    """
    Inicjalizuje i keszuje pipeline do klasyfikacji obrazów.
    Używa GPU (CUDA/MPS), jeśli dostępne, w przeciwnym razie CPU.
    """
    # wybór urządzenia: CUDA > MPS (Apple) > CPU
    device = -1
    if torch.cuda.is_available():
        device = 0
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = -1

    return pipeline(
        task="image-classification",
        model=model_id,
        device=device,
    )