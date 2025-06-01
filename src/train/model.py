import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model(model_path):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 7)
)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]), 
])

def predict(model, face_image: Image.Image):
    img_tensor = transform(face_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.Softmax(dim=1)(outputs)
        predicted_class = torch.argmax(probs, 1).item()
        predicted_emotion = classes[predicted_class]
        probabilities = probs.cpu().numpy().flatten()
    return predicted_emotion, probabilities
