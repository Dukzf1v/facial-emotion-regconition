import streamlit as st
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
import pandas as pd
import torch.nn as nn
import cv2
import io
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights

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
model.load_state_dict(torch.load(r"D:\StudyPath\GR1\Facial-Emotion-Regconition\model\best_model.pth", map_location=device))

model.eval()

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(r'D:\StudyPath\GR1\Facial-Emotion-Regconition\src\haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

st.title("Emotion Recognition")

uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))  
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid image file.")
    else:
        image_rgb = image.convert('RGB')  
        image_np = np.array(image_rgb) 

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            all_probabilities = [] 
            all_emotions = []  

            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_crop = image_np[y:y + h, x:x + w]
                pil_face = Image.fromarray(face_crop) 
                img_tensor = transform(pil_face).unsqueeze(0).to(device) 

                with torch.no_grad():
                    outputs = model(img_tensor) 
                    probabilities = torch.nn.Softmax(dim=1)(outputs)  
                    predicted_class = torch.argmax(probabilities, 1) 
                    predicted_emotion = classes[predicted_class.item()]  

                    probabilities = probabilities.cpu().numpy().flatten()  
                    all_probabilities.append(probabilities)  
                    all_emotions.append(predicted_emotion)  

                cv2.putText(image_np, f'{i + 1}:{predicted_emotion}-{probabilities[predicted_class.item()] * 100:.2f}% ', 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            image_with_boxes = Image.fromarray(image_np)
            st.image(image_with_boxes, caption='Processed Image with Emotion and Bounding Box', use_container_width=True)

            probabilities_df = []

            for i, emotion in enumerate(all_emotions):
                data = {
                    'Face Index': f'Face {i + 1}',
                    **{classes[j]: f'{prob*100:.2f}%' for j, prob in enumerate(all_probabilities[i])} 
                }
                probabilities_df.append(data)

            result_df = pd.DataFrame(probabilities_df)
            st.write("Class Probabilities for Detected Faces:", result_df)
        else:
            st.write("No faces detected in the image.")
