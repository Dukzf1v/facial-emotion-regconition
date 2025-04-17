from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import shutil

train_root = r'D:\StudyPath\GR1\Facial-Emotion-Regconition\fer2013_cropped\train'
new_train_root = r'D:\StudyPath\GR1\Facial-Emotion-Regconition\new_fer2013_cropped\train'
val_root = r'D:\StudyPath\GR1\Facial-Emotion-Regconition\new_fer2013_cropped\val'
os.makedirs(new_train_root, exist_ok=True)
os.makedirs(val_root, exist_ok=True)

for cls in tqdm(os.listdir(train_root)):
    cls_path = os.path.join(train_root, cls)
    images = os.listdir(cls_path)

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(new_train_root, cls), exist_ok=True)
    os.makedirs(os.path.join(val_root, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(new_train_root, cls, img))

    for img in val_imgs:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(val_root, cls, img))

src_test_root = r'D:\StudyPath\GR1\Facial-Emotion-Regconition\fer2013_cropped\test'
dst_test_root = r'D:\StudyPath\GR1\Facial-Emotion-Regconition\new_fer2013_cropped\test'
os.makedirs(dst_test_root, exist_ok=True)

for cls in os.listdir(src_test_root):
    os.makedirs(os.path.join(dst_test_root, cls), exist_ok=True)
    for img in os.listdir(os.path.join(src_test_root, cls)):
        shutil.copy2(
            os.path.join(src_test_root, cls, img),
            os.path.join(dst_test_root, cls, img)
        )

