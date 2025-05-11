import cv2
import torch
import numpy as np
import torch.nn as nn

from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights
from PIL import Image

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 4096),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 7)
)
model.to(device)  

model.load_state_dict(torch.load(r"..\model\best_model.pth", map_location=device))
model.eval()  

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

def draw_probabilities_bar(frame, probabilities):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    start_x = frame.shape[1] - 200 
    start_y = 30  

    for i, prob in enumerate(probabilities):
        label = f"{emotion_labels[i]}: {prob*100:.2f}%" 
        cv2.putText(frame, label, (start_x, start_y + (i * 30)), font, 0.7, (0, 255, 0), 2)

    return frame

cv2.namedWindow('Emotion Recognition', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Emotion Recognition', cv2.WND_PROP_FULLSCREEN, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(pil_face).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)  
            probabilities = torch.nn.Softmax(dim=1)(outputs)  
            predicted_class = torch.argmax(probabilities, 1) 
            predicted_emotion = emotion_labels[predicted_class.item()] 

        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 

        frame = draw_probabilities_bar(frame, probabilities.cpu().numpy().flatten())

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()