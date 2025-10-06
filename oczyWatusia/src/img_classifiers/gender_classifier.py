from PIL import Image
from .utils import _get_pipe

def classify_gender(img):
    pipe = _get_pipe(model_id="rizvandwiki/gender-classification-2")

    if not isinstance(img, Image.Image):
        raise TypeError("Oczekiwano obiektu PIL.Image.Image")

    # Upewnij się, że obraz jest w formacie RGB (częste wymaganie procesora obrazów)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    return pipe(img)[0]["label"]
