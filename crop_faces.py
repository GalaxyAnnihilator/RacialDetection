import cv2
import os

# Load OpenCV's Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define paths
RAW_DATA_DIR = "rawdata"  # Adjusted to match your structure
OUTPUT_DIR = "processed_faces"  # New folder for processed images

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def detect_and_crop_faces():
    for category in os.listdir(RAW_DATA_DIR):  # Loop through racial categories
        input_folder = os.path.join(RAW_DATA_DIR, category)
        output_folder = os.path.join(OUTPUT_DIR, category)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith((".jpg", ".jpeg")):
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Error loading image: {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for i, (x, y, w, h) in enumerate(faces):
                    face = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (128, 128))  # Resize to 128x128
                    output_path = os.path.join(output_folder, f"{category}_{i}_{filename}")
                    cv2.imwrite(output_path, face_resized)
                    print(f"Saved: {output_path}")

detect_and_crop_faces()
