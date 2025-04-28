import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from resnet import ResNet, ResidualBlock

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

model = ResNet(
    residual_block=ResidualBlock,
    n_blocks_lst=[2, 2, 2, 2],
    n_classes=7
).to(device)

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

model.load_state_dict(torch.load(r"..\model\best_model_resnet18.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),          
    transforms.ToTensor(),                
    transforms.Normalize([0.456], [0.224])   
])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

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
            _, predicted_class = torch.max(outputs, 1)
            predicted_emotion = emotion_labels[predicted_class.item()]

        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
