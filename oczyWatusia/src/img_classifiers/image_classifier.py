import time

import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification, ViTImageProcessor, \
    ViTForImageClassification

from oczyWatusia.src.img_classifiers import  findDominantColor
from oczyWatusia.src.img_classifiers.utils import retrieve_img


class ImageClassifier:
    def __init__(self, idx: str):
        self.id = idx
        self.model = None
        self.processor = None
        self.id2label = {}

    def load_models(self):
        self.processor = AutoImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = SiglipForImageClassification.from_pretrained(self.id)

    def process(self, img):
        img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        prediction = {
            self.id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
        }
        return prediction

class EmotionClassifier(ImageClassifier):
    def __init__(self):
        super().__init__('abhilash88/face-emotion-detection')
        self.id2label = {
            "0": "Angry",
            "1": "Disgust",
            "2": "Fear",
            "3": "Happy",
            "4": "Sad",
            "5": "Surprise",
            "6": "Neutral",
        }

    def load_models(self):
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)

class GenderClassifier(ImageClassifier):
    def __init__(self):
        super().__init__("prithivMLmods/Realistic-Gender-Classification")
        self.id2label = {
            "0": "female",
            "1": "male"
        }

class AgeClassifier(ImageClassifier):
    def __init__(self):
        super().__init__("prithivMLmods/open-age-detection")
        self.id2label = {
            "0": "Child 0-12",
            "1": "Teenager 13-20",
            "2": "Adult 21-44",
            "3": "Middle Age 45-64",
            "4": "Aged 65+"
        }

def getClassifiers():
    emotion_classifier = EmotionClassifier()
    gender_classifier = GenderClassifier()
    age_classifier = AgeClassifier()
    emotion_classifier.load_models()
    gender_classifier.load_models()
    age_classifier.load_models()
    return emotion_classifier, gender_classifier, age_classifier

if __name__ == "__main__":
    img = retrieve_img("https://wassyl.pl/hpeciai/849585a6d57bcc5edbf17f94ccec1d35/pol_pl_Viralowa-Dresowa-bawelniana-BLUZA-z-kapturem-oversize-szeroka-baby-pink-E644-14377_1.jpg")
    emotion_classifier = EmotionClassifier()
    gender_classifier = GenderClassifier()
    age_classifier = AgeClassifier()
    emotion_classifier.load_models()
    gender_classifier.load_models()
    age_classifier.load_models()

    start = time.time()
    print("Uploaded!")
    color = findDominantColor(np.asarray(img))
    print(color, time.time() - start)

    emotion = emotion_classifier.process(img)
    print(emotion, time.time() - start)

    gender = gender_classifier.process(img)
    print(gender, time.time() - start)

    age = age_classifier.process(img)
    print(age, time.time() - start)


