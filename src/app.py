import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from resnet import ResNet
import torch.nn.functional as F
from mtcnn import MTCNN

# Device setup
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN with the device name as a string
mtcnn = MTCNN(device=device_name)  # Using device as a string

# Initialize and load the ResNet model
device = torch.device(device_name)  # Set the PyTorch device (same as MTCNN)
model = ResNet(n_classes=7)  # Initialize the emotion detection model
model.load_state_dict(torch.load("D:/StudyPath/GR1/Facial Emotion Regconition/model/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Emotion classes
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to 224x224
    img = np.array(image)  # Convert to numpy array
    if len(img.shape) == 2:  # If the image is grayscale
        img = np.expand_dims(img, axis=-1)  # Add an extra channel dimension
        img = np.repeat(img, 3, axis=-1)  # Repeat the grayscale channel to create 3 channels
    img = torch.tensor(img).permute(2, 0, 1).float()  # Convert to tensor and reorder dimensions
    normalized_img = img / 255.0  # Normalize image to [0, 1]
    return normalized_img

# Streamlit UI
st.title("Emotion Recognition")
st.write("Using your webcam, the model will detect your emotion in real-time.")

cap = cv2.VideoCapture(0)  # Open webcam

# Create an empty placeholder for displaying images
frame_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

    # Detect faces using MTCNN
    faces = mtcnn.detect_faces(frame_rgb)  # Use detect_faces to detect faces

    if faces is not None:
        # Process each detected face
        for face in faces:
            # Get bounding box and crop the face region
            x, y, w, h = face['box']  # Get bounding box coordinates
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw the bounding box

            # Crop and preprocess the face region
            face_crop = frame_rgb[y:y+h, x:x+w]  # Crop face region
            pil_face = Image.fromarray(face_crop)  # Convert to PIL image
            img_tensor = preprocess_image(pil_face).unsqueeze(0).to(device)  # Preprocess image

            # Model inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predicted_class_name = classes[predicted_class]
                prob_dict = {classes[i]: probabilities[0][i].item() for i in range(len(classes))}

            # Annotate the frame with the predicted emotion
            cv2.putText(frame_rgb, f'Emotion: {predicted_class_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display predicted emotion and probabilities in Streamlit
            st.subheader(f"Predicted Emotion: {predicted_class_name}")
            st.write("Probabilities for each emotion:")
            st.write(prob_dict)

    # Convert back to BGR for OpenCV display
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Update the Streamlit frame placeholder with the new image
    frame_placeholder.image(frame_bgr, channels="BGR", use_container_width=True)  # Use use_container_width

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
