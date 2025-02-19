import os
import csv

PROCESSED_DIR = "processed_faces"
LABELS_CSV = "labels.csv"

LABEL_MAP = {
    "white": 0,
    "black": 1,
    "asian": 2,
    "latino": 3,
    "nativeamerican": 4,
    "pacificislander": 5,
    "middleeast": 6
}

def clean_label(label):
    return "_".join(label.split("_")[1:])  # Removes number prefix

def create_labels_csv():
    with open(LABELS_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "label"])

        for category in os.listdir(PROCESSED_DIR):
            category_path = os.path.join(PROCESSED_DIR, category)

            if not os.path.isdir(category_path):
                continue

            cleaned_label = clean_label(category)
            numeric_label = LABEL_MAP.get(cleaned_label, -1)  # Convert to number

            for filename in os.listdir(category_path):
                writer.writerow([filename, numeric_label])

    print(f"Labels CSV updated with numeric mapping: {LABELS_CSV}")

create_labels_csv()
