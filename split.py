import os
import shutil
import random

PROCESSED_DIR = "processed_faces"
OUTPUT_DIR = "dataset"  # New directory for train/test split
TRAIN_RATIO = 0.8  # 80% training, 20% testing

def split_dataset():
    for category in os.listdir(PROCESSED_DIR):
        category_path = os.path.join(PROCESSED_DIR, category)

        if not os.path.isdir(category_path):
            continue

        images = os.listdir(category_path)
        random.shuffle(images)  # Shuffle for randomness

        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for split, split_images in [("train", train_images), ("test", test_images)]:
            split_folder = os.path.join(OUTPUT_DIR, split, category)
            os.makedirs(split_folder, exist_ok=True)

            for img in split_images:
                src_path = os.path.join(category_path, img)
                dest_path = os.path.join(split_folder, img)
                shutil.move(src_path, dest_path)

    print(f"Dataset split into {OUTPUT_DIR}/train and {OUTPUT_DIR}/test")

split_dataset()
