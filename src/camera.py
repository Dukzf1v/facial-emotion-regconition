import cv2
import os 
import numpy as np
from PIL import Image
from train.model import load_model, predict, classes

cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoint', 'best_model.pth')
model = load_model(model_path)

def draw_probabilities_bar(frame, probabilities):
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_x = frame.shape[1] - 220
    start_y = 30

    for i, prob in enumerate(probabilities):
        label = f"{classes[i]}: {prob*100:.2f}%"
        cv2.putText(frame, label, (start_x, start_y + i*30), font, 0.7, (0, 255, 0), 2)

    return frame

cv2.namedWindow('Emotion Recognition', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Emotion Recognition', cv2.WND_PROP_FULLSCREEN, 1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        predicted_emotion, probabilities = predict(model, pil_face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame = draw_probabilities_bar(frame, probabilities)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
