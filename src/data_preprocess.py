import os
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np 

faceCascade = cv2.CascadeClassifier('D:\\StudyPath\\GR1\\Facial-Emotion-Regconition\\src\\haarcascade_frontalface_default.xml')

input_root = 'D:\\StudyPath\\GR1\\Facial-Emotion-Regconition\\data'
output_root = 'D:\\StudyPath\\GR1\\Facial-Emotion-Regconition\\kdef_cropped'

os.makedirs(output_root, exist_ok=True)
for cls in os.listdir(input_root):
    os.makedirs(os.path.join(output_root, cls), exist_ok=True)

def crop_and_save_face(image_path, output_path):
    try:
        img = Image.open(image_path).convert('RGB')  
        img_np = np.array(img)  
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            return False  

        x, y, w, h = faces[0]
        x, y = max(0, x), max(0, y)
        
        face = img.crop((x, y, x + w, y + h)).resize((160, 160))
        face.save(output_path) 
        return True

    except Exception as e:
        print(f"[ERROR] {image_path} → {e}")
        return False

for cls in os.listdir(input_root):
    input_folder = os.path.join(input_root, cls)
    output_folder = os.path.join(output_root, cls)

    for img_name in tqdm(os.listdir(input_folder), desc=f'{cls}'):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        crop_and_save_face(input_path, output_path)


