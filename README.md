# **Facial Emotion Recognition (FER) using ResNet18**

## **Overview**
This project leverages a **ResNet-based deep learning model** for **Facial Emotion Recognition**. The system detects faces using **MTCNN**, crops them, and classifies the emotions from the faces into 7 categories:  
**Anger**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

---

## **Installation**

### **Prerequisites:**

1. Clone the repository:
    ```bash
    git clone https://github.com/Dukzf1v/facial-emotion-regconition.git
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run `control.py` to use the real-time model:
    ```bash
    python -u control.py
    ```

4. Run `app.py` to launch the Streamlit web interface:
    ```bash
    streamlit run app.py
    ```

---

## **Dataset: FER2013**

This model is trained using the **FER2013** dataset, which is a publicly available dataset containing facial expressions of different emotions.

---

## **Model** 

### **Hyperparameters:**
- **Batch Size**: 32
- **Optimizer**:
  - Learning rate: 0.001
  - Momentum: 0.9
  - Weight Decay: 0.0001
- **Number of epochs**: 55
- **Early Stop**: 
  - Patience: 8

### **Model Performance**:
- **Val accuracy:**  0.6569363358354184 / **Test accuracy:**  0.6555295689943201
  
- **Train/Validation Loss/Accuracy**:
  
  <img src="https://github.com/user-attachments/assets/0173cd12-1fdc-46ff-a631-9ae54321952d" width="300" style="display:inline-block; margin-right:10px;">

- **Precision, Recall, F1**:
  
  <img src="https://github.com/user-attachments/assets/bcb8174d-b6d7-4981-913d-8079ceb753ce" width="300" style="display:inline-block; margin-right:10px;">

- **Confusion Matrix**:
  
  <img src="https://github.com/user-attachments/assets/d1df6ebb-8750-46d6-b7f2-de657b4c0a8d" width="300" style="display:inline-block;">
---

## **Results**:
- **Output Image**: [View output images here](https://github.com/Dukzf1v/facial-emotion-regconition/tree/6912a5a4d5e2757a6bffa5d5b8907a5e29d7aa25/output%20image)

---

<img src="https://github.com/user-attachments/assets/706cc52b-202c-4f99-ac9a-a260db88e88f" width="200" style="display:inline-block; margin-right:10px;">
<img src="https://github.com/user-attachments/assets/0633c47c-7988-4724-9237-c772a694c613" width="200" style="display:inline-block; margin-right:10px;">
