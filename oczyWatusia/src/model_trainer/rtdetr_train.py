import os
import torch
import supervision as sv
import albumentations as A
import wandb

from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from evaluator import MAPEvaluator
from dotenv import load_dotenv
load_dotenv()

from data_loader import ds_test, ds_valid, ds_train, id2label, label2id

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
CHECKPOINT = os.environ.get("CHECKPOINT")
FT_MODEL_DIR = os.environ.get("FT_MODEL_DIR")

def collate_fn(batch):
    data = {"pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": [x["labels"] for x in batch]}
    return data


class PyTorchDetectionDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    @staticmethod
    def annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, image, annotations = self.dataset[idx]

        # Convert image to RGB numpy array
        image = image[:, :, ::-1].copy()
        boxes = annotations.xyxy
        categories = annotations.class_id

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category=categories
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]


        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=categories, boxes=boxes)
        result = self.processor(
            images=image, annotations=formatted_annotations, return_tensors="pt")

        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    IMAGE_SIZE = 480

    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT,
        do_resize=True,
        use_fast=True,
        size={"width": IMAGE_SIZE, "height": IMAGE_SIZE},
    )
    train_augmentation_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            clip=True,
            min_area=25
        ),
    )

    valid_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            clip=True,
            min_area=1
        ),
    )

    pytorch_dataset_train = PyTorchDetectionDataset(
        ds_train, processor, transform=train_augmentation_and_transform)
    pytorch_dataset_valid = PyTorchDetectionDataset(
        ds_valid, processor, transform=valid_transform)
    pytorch_dataset_test = PyTorchDetectionDataset(
        ds_test, processor, transform=valid_transform)


    eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.01, id2label=id2label)
    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=f"finetune",
        num_train_epochs=10,
        max_grad_norm=0.1,
        learning_rate=5e-5,
        warmup_steps=300,
        per_device_train_batch_size=2,
        dataloader_num_workers=0,
        metric_for_best_model="eval_map_50_95",
        greater_is_better=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=["none"],
        logging_strategy="steps",
        logging_steps=500,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    wandb.login(key=WANDB_API_KEY)
    trainer.train()

    model.save_pretrained(FT_MODEL_DIR)
    processor.save_pretrained(FT_MODEL_DIR)

    targets = []
    predictions = []

    for i in range(len(ds_test)):
        path, _, annotations = ds_test[i]

        image = Image.open(path)
        inputs = processor(image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        w, h = image.size
        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.3)

        detections = sv.Detections.from_transformers(results[0])

        targets.append(annotations)
        predictions.append(detections)

    # @title Calculate mAP
    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )

    print(f"map50_95: {mean_average_precision.map50_95:.2f}")
    print(f"map50: {mean_average_precision.map50:.2f}")
    print(f"map75: {mean_average_precision.map75:.2f}")

    # @title Calculate Confusion Matrix
    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=ds_test.classes
    )

    confusion_matrix.plot()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()