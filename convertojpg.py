from PIL import Image
import os

RAW_DATA_DIR = "rawdata"  # Your main dataset folder

def convert_to_jpg():
    for category in os.listdir(RAW_DATA_DIR):  # Loop through racial categories
        category_path = os.path.join(RAW_DATA_DIR, category)
        
        if not os.path.isdir(category_path):  # Skip files
            continue

        for filename in os.listdir(category_path):
            if filename.lower().endswith(".jpeg"):  # Only convert .jpeg
                file_path = os.path.join(category_path, filename)
                img = Image.open(file_path)
                new_filename = filename.replace(".jpeg", ".jpg")
                new_path = os.path.join(category_path, new_filename)
                
                img.convert("RGB").save(new_path, "JPEG")  # Convert and save
                os.remove(file_path)  # Delete the old .jpeg file
                print(f"Converted: {file_path} → {new_path}")

convert_to_jpg()
