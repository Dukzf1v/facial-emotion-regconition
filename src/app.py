import streamlit as st
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch.nn.functional as F
import cv2
import io
from resnet import ResNet, ResidualBlock
from torchvision import transforms
import os
st.write("Current working directory:", os.getcwd())

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

model = ResNet(
    residual_block=ResidualBlock,
    n_blocks_lst=[2, 2, 2, 2],
    n_classes=7
).to(device)

model.load_state_dict(torch.load(r"/mount/src/fer/model/best_model_resnet18.pth", map_location=device))

model.eval()

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(r'/mount/src/fer/src/haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  
])

st.title("Emotion Recognition")

uploaded_file = st.file_uploader("Choose Image",type=["jpg", "jpeg", "png"])

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
            for (x, y, w, h) in faces:
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

                face_crop = image_np[y:y + h, x:x + w] 
                pil_face = Image.fromarray(face_crop) 
                img_tensor = transform(pil_face).unsqueeze(0).to(device) 

                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, predicted_class = torch.max(outputs, 1)
                    predicted_emotion = classes[predicted_class.item()]

                cv2.putText(image_np, f'{predicted_emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            image_with_boxes = Image.fromarray(image_np)
            st.image(image_with_boxes, caption='Processed Image with Emotion and Bounding Box', use_container_width=True)

        else:
            st.write("No faces detected in the image.")
