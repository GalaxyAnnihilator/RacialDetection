import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class SimpleVGG(nn.Module):
    def __init__(self, num_classes):
        super(SimpleVGG, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define labels (Make sure these match your training classes)
class_names = ["white", "black", "asian", "latino", "nativeamerican", "pacificislander", "middleeast"]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match training size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

def crop_face(image_path):
    """Detect and crop the face from an image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("⚠️ No face detected, using the full image!")
        return Image.open(image_path).convert("RGB")  # Return the original image

    # Select the first detected face
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]  # Crop the face region

    # Convert OpenCV image (BGR) to PIL image (RGB)
    return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

def load_model(model_path="simple_vgg.pth"):
    """Load the entire saved model."""
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()  # Set to evaluation mode
    return model

def predict(image_path="test.jpg", model_path="simple_vgg.pth"):
    """Predict the class of an image using a saved model."""
    model = load_model(model_path)

    # Detect and crop the face
    face_image = crop_face(image_path)
    image = transform(face_image).unsqueeze(0)  # Preprocess image

    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Print prediction result
    print(f"Predicted Class: {class_names[predicted_class]} ({probabilities[0][predicted_class]:.2f} confidence)")
    return class_names[predicted_class], round(probabilities[0][predicted_class].item() * 100)

if __name__ == "__main__":
    predict()