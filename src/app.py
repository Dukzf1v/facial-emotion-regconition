import streamlit as st
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError 
from resnet import ResNet
import torch.nn.functional as F
from mtcnn import MTCNN
import io
import cv2

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(device=device_name) 

device = torch.device(device_name)  
model = ResNet(n_classes=7) 
model.load_state_dict(torch.load("D:/StudyPath/GR1/Facial Emotion Regconition/model/best_model.pth", map_location=device))
model.to(device)
model.eval()

classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def preprocess_image(image):
    image = image.resize((224, 224)) 
    img = np.array(image)  
    if len(img.shape) == 2:  
        img = np.expand_dims(img, axis=-1) 
        img = np.repeat(img, 3, axis=-1)  
    img = torch.tensor(img).permute(2, 0, 1).float() 
    normalized_img = img / 255.0  
    return normalized_img

st.title("Emotion Recognition from Uploaded Image")
st.write("Upload an image, and the model will detect faces and predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image_bytes = uploaded_file.read()
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid image file.")
    else:

        image_rgb = image.convert('RGB')
        image_np = np.array(image_rgb)

        faces = mtcnn.detect_faces(image_np) 

        if faces is not None:

            for face in faces:

                x, y, w, h = face['box'] 
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2) 

                face_crop = image_np[y:y+h, x:x+w] 
                pil_face = Image.fromarray(face_crop) 
                img_tensor = preprocess_image(pil_face).unsqueeze(0).to(device) 

                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    predicted_class_name = classes[predicted_class]
                    prob_dict = {classes[i]: probabilities[0][i].item() for i in range(len(classes))}

                cv2.putText(image_np, f'{predicted_class_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
            image_with_boxes = Image.fromarray(image_np)
            st.image(image_with_boxes, caption='Processed Image with Emotion and Bounding Box', use_container_width=True)
        else:
            st.write("No faces detected in the image.")
