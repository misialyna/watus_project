import os
from pathlib import Path

import numpy as np
import torch
import supervision as sv
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from ultralytics.data import YOLODataset

DS_PATH = Path(os.environ.get("DS_PATH"))
DS_NAME = "clothesYOLO"
DS_PATH = DS_PATH / DS_NAME

class CocoDetAsTargets(Dataset):
    """
    Owija torchvision.datasets.CocoDetection i zwraca:
      image: Tensor[C,H,W]
      target: dict(boxes=Tensor[N,4], labels=Tensor[N]) w formacie xyxy.
    Filtruje boksy o zerowej/ujemnej szer./wys. i klamruje do granic obrazu.
    """
    def __init__(self, root: str, ann_file: str, tf=None):
        self.inner = datasets.CocoDetection(root=root, annFile=ann_file, transform=tf)
        self.coco = self.inner.coco

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        img, anns = self.inner[idx]

        # rozmiar obrazu
        if isinstance(img, torch.Tensor):
            h, w = int(img.shape[-2]), int(img.shape[-1])
        else:
            w, h = img.size  # PIL: (W,H)

        boxes_list, labels_list = [], []
        for a in anns:
            x, y, ww, hh = a["bbox"]  # COCO: xywh
            if ww is None or hh is None or ww <= 0 or hh <= 0:
                continue
            x1, y1 = x, y
            x2, y2 = x + ww, y + hh

            # klamrowanie do [0, W-1]/[0, H-1]
            x1 = max(0.0, min(float(x1), w - 1.0))
            y1 = max(0.0, min(float(y1), h - 1.0))
            x2 = max(0.0, min(float(x2), w - 1.0))
            y2 = max(0.0, min(float(y2), h - 1.0))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes_list.append([x1, y1, x2, y2])
            labels_list.append(int(a["category_id"]))

        if boxes_list:
            boxes = torch.tensor(boxes_list, dtype=torch.float32)
            labels = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target

tf = transforms.Compose([transforms.ToTensor()])
dataset = CocoDetAsTargets(str(DS_PATH / 'train'),
                           str(DS_PATH / 'train/_annotations.coco.json'),
                            tf=tf)

ds_train = sv.DetectionDataset.from_coco(
    images_directory_path=str(DS_PATH / 'train'),
    annotations_path=str(DS_PATH / 'train/_annotations.coco.json'),
)
ds_valid = sv.DetectionDataset.from_coco(
    images_directory_path=str(DS_PATH / 'valid'),
    annotations_path=str(DS_PATH / 'valid/_annotations.coco.json'),
)
ds_test = sv.DetectionDataset.from_coco(
    images_directory_path=str(DS_PATH / 'test'),
    annotations_path=str(DS_PATH / 'test/_annotations.coco.json'),
)

id2label = {id: label for id, label in enumerate(ds_train.classes)}
label2id = {label: id for id, label in enumerate(ds_train.classes)}

print(f"Number of training images: {len(ds_train)}")
print(f"Number of validation images: {len(ds_valid)}")
print(f"Number of test images: {len(ds_test)}")

GRID_SIZE = 5

def annotate(image, annotations, classes):
    labels = [
        classes[class_id]
        for class_id
        in annotations.class_id
    ]

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels=labels)
    return annotated_image

