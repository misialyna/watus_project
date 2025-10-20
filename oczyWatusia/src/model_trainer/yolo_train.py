from datetime import datetime
import os
import shutil

import cv2
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import ultralytics.data.build as dataset

from yolo_dataset import YOLOWeightedDataset

load_dotenv()
if os.environ.get("USE_EXPERIMENTAL_BALANCER", "0") == "1":
    print("!Using experimental balancer!")
    dataset.YOLODataset = YOLOWeightedDataset

from ultralytics.models import YOLO
# from wandb.integration.ultralytics import add_wandb_callback
# import wandb

ds_dir = "./datasets/finale-6"
data_yaml_path = ds_dir + "/data.yaml"
test_path=ds_dir + "/test"
base_dir = "./src/model_trainer/"
model_save_dir = base_dir + "/ft-yolo-models"
base_model_name = "yolo12s"
projectName = "OczyWatusia"
logs_dir = "training_logs"

def create_validation_set(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir):
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(train_images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    np.random.seed(42)
    np.random.shuffle(image_files)
    val_split = int(0.2 * len(image_files))
    val_files = image_files[:val_split]

    for img_file in val_files:
        # Copy image
        src_img = os.path.join(train_images_dir, img_file)
        dst_img = os.path.join(val_images_dir, img_file)
        shutil.copy2(src_img, dst_img)

        # Copy corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(train_labels_dir, label_file)
        dst_label = os.path.join(val_labels_dir, label_file)

        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)


def train_yolo_model(epochs=30, batch_size=4, img_size=640, lr0=0.01):
    # Check for CUDA availability
    device = '0' if torch.cuda.is_available() else 'cpu'

    # Define timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'train_{timestamp}'

    # Load the model
    try:
        model = YOLO(f"{base_model_name}.pt")
        model_type = base_model_name
    except Exception:
        model = YOLO('yolov8n.pt')
        model_type = 'yolov8n'

    WANDB_API_KEY=os.environ.get('WANDB_API_KEY')
    # wandb.login(key=WANDB_API_KEY)
    # wandb.init(project=projectName, name=run_name, job_type='train')
    # add_wandb_callback(model, enable_model_checkpointing=True)
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        weight_decay=0.001,
        close_mosaic=0,
        dropout=0.01,
        patience=10,
        save=True,
        device=device,
        project=logs_dir,
        name=run_name,
        lr0=lr0,
        lrf=0.01,
        plots=True,
        save_period=5,
        freeze=18,
        workers=8,
        classes=[
            0,
            1,
            2,
            # 3,
            4,
            5,
            6,
            7,
            8,
            9,
            # 10,
            11,
            12,
            13,
            # 14
        ],
        # augment=True,
        # translate=0.1,
        # scale=0.1,
        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.4,
    )
    # Save the model
    model_save_path = os.path.join(model_save_dir, f"{model_type}_{timestamp}.pt")

    try:
        model.save(model_save_path)
    except AttributeError:
        try:
            model.save(model_save_path)
        except Exception:
            best_model_path = os.path.join(base_dir, 'runs', run_name, 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_save_path)
    # finally:
    #     wandb.finish()
    return model


def validate_model(model):
    metrics = model.val(
        data=data_yaml_path,
        split='val',
        project=logs_dir,
        name='val'
    )

    # Calculate F1 score
    f1_score = 2 * metrics.box.p * metrics.box.r / (
                metrics.box.p + metrics.box.r + 1e-6)
    print(f"F1 score: {f1_score}")

    return metrics


def test_on_images(model, conf_threshold=0.25):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, 'runs', f'detect_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    test_images_dir = os.path.join(test_path, "images")
    image_files = [f for f in os.listdir(test_images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    with open(data_yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    class_names = yaml_data.get('names', ['Unknown'])

    # Process a subset of images for visualization
    viz_images = image_files[:min(10, len(image_files))]

    for img_file in viz_images:
        img_path = os.path.join(test_images_dir, img_file)

        # Run inference
        results = model(img_path, conf=conf_threshold)

        # Get original image for visualization
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw boxes on image
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Get class name
            class_name = class_names[cls] if cls < len(class_names) else f"Unknown-{cls}"

            # Generate a unique color for each class
            color = ((cls * 70) % 256, (cls * 50) % 256, (cls * 30) % 256)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save annotated image
        output_path = os.path.join(output_dir, img_file)
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

train_path = ds_dir + '/train'
val_path = ds_dir + '/val'
class_names = ['T-shirt', 'aifh', 'boy', 'cros', 'dress', 'girl', 'objects', 'shorts', 'skirt', 'sweater', 'trousers']

def verify_dataset_structure():
    # Check if data.yaml exists and create if needed
    if not os.path.exists(data_yaml_path):
        train_images_path = os.path.join(train_path, "images")
        val_images_path = os.path.join(val_path, "images")
        test_images_path = os.path.join(test_path, "images")

        yaml_data = {
            'train': train_images_path,
            'val': val_images_path,
            'test': test_images_path,
            'nc': len(class_names),
            'names': class_names
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

    # Count and verify images and labels
    train_images_dir = os.path.join(train_path, "images")
    train_labels_dir = os.path.join(train_path, "labels")

    train_images = len([f for f in os.listdir(train_images_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    train_labels = len([f for f in os.listdir(train_labels_dir) if f.endswith('.txt')])

    return train_images > 0 and train_labels > 0

def main():
    if not verify_dataset_structure():
        return

    model = train_yolo_model()

    if model is not None:
        validate_model(model)
        test_on_images(model)
if __name__ == "__main__":
    main()