import cv2
import io
import os 
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from train.model import load_model, predict, classes

cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoint', 'best_model.pth')
model = load_model(model_path)

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
                predicted_emotion, probabilities = predict(model, pil_face)
                all_probabilities.append(probabilities)
                all_emotions.append(predicted_emotion)

                cv2.putText(image_np,
                            f'{i + 1}:{predicted_emotion}-{probabilities[np.argmax(probabilities)] * 100:.2f}%',
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            image_with_boxes = Image.fromarray(image_np)
            st.image(image_with_boxes, caption='Processed Image', use_container_width=True)

            probabilities_df = []
            for i, emotion in enumerate(all_emotions):
                data = {
                    'Face Index': f'Face {i + 1}',
                    **{cls: f'{prob*100:.2f}%' for cls, prob in zip(classes, all_probabilities[i])}
                }
                probabilities_df.append(data)

            result_df = pd.DataFrame(probabilities_df)
            st.write("Class Probabilities for Detected Faces:", result_df)
        else:
            st.write("No faces detected in the image.")
