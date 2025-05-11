# **Facial Emotion Recognition (FER) using ResNet18**

## **Overview**
This project uses a **ResNet-based deep learning model** for **Facial Emotion Recognition**. The system detects faces using **MTCNN**, crops them, and then classifies the emotions from the faces into 7 categories: **Anger**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

## **Installation**

### Prerequisites:

Clone repository:

```bash
git clone https://github.com/Dukzf1v/facial-emotion-regconition.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

- Run `control.py` to use the real-time model:
  ```bash
  python -u control.py
  ```
- Run `app.py` to use streamlit web
  ```bash
  streamlit run app.py
  ```
## Dataset: FER2013

## Model: 
* Hyperparameters:
  - Batch Size: 32
  - Optimizer: + Learning rate: 0.001
               + Momentum: 0.9
               + Weight_decay: 0.0001
  - Number of epochs: 55
  - Early Stop: + Patience: 8
* Train/Val Loss/Accuracy:
  ![image](https://github.com/user-attachments/assets/0173cd12-1fdc-46ff-a631-9ae54321952d)

* Precision, Recall, F1:
  ![image](https://github.com/user-attachments/assets/bcb8174d-b6d7-4981-913d-8079ceb753ce)

* Confusion Matrix:
  ![image](https://github.com/user-attachments/assets/d1df6ebb-8750-46d6-b7f2-de657b4c0a8d)

## Result:
- Output image: https://github.com/Dukzf1v/facial-emotion-regconition/tree/6912a5a4d5e2757a6bffa5d5b8907a5e29d7aa25/output%20image
<img src="https://github.com/user-attachments/assets/706cc52b-202c-4f99-ac9a-a260db88e88f" width="200" style="display:inline-block; margin-right:10px;">
<img src="https://github.com/user-attachments/assets/0633c47c-7988-4724-9237-c772a694c613" width="200" style="display:inline-block;">




