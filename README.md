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
- **Batch Size**: 64
- **Optimizer**:
  - Learning rate: 0.001
  - Momentum: 0.9
  - Weight Decay: 0.0001
- **Number of epochs**: 100
- **Scheduler**: 
  - CosineAnnealingLR

### **Model Performance**:
- **Val accuracy:**  0.6569363358354184
  
- **Train/Validation Loss/Accuracy**:
  
  <img src="https://github.com/user-attachments/assets/31559170-db41-4e69-9465-cbd8a6c17b89" width="300" style="display:inline-block; margin-right:10px;">

### **Evaluation**:
#### **Fer2013 Test**:
- **Precision, Recall, F1**:

  <img src="https://github.com/user-attachments/assets/f345bde3-1f99-44aa-9b08-a2c9a194e145" width="300" style="display:inline-block; margin-right:10px;">

- **Confusion Matrix**:

  <img src="https://github.com/user-attachments/assets/41ed1cce-f2d5-4467-b6b3-39b52ce7526d" width="300" style="display:inline-block;">

#### **KDEF**:
- **Precision, Recall, F1**:

  <img src="https://github.com/user-attachments/assets/81e27d1b-669e-4d30-a041-5600899df779" width="300" style="display:inline-block; margin-right:10px;">

- **Confusion Matrix**:

  <img src="https://github.com/user-attachments/assets/4317b59f-ca94-403a-b7bf-5f08ad0032b4" width="300" style="display:inline-block;">
  
#### **Raf-db**:
- **Precision, Recall, F1**:

  <img src="https://github.com/user-attachments/assets/2b85eede-d3c4-4715-a2e4-67b33c9caa31" width="300" style="display:inline-block; margin-right:10px;">

- **Confusion Matrix**:

  <img src="https://github.com/user-attachments/assets/b49305fb-0c14-4a78-a4c4-36d451f397d9" width="300" style="display:inline-block;">

---
## **Results**:
- Linh streamlit: https://1convitxoera2caicanh.streamlit.app/
---
- **Output**: [View output here](https://github.com/Dukzf1v/facial-emotion-regconition/tree/6912a5a4d5e2757a6bffa5d5b8907a5e29d7aa25/output%20image)
---

<img src="https://github.com/Dukzf1v/facial-emotion-regconition/blob/c97e7c4b812221984870af9305e15e99e6251085/output%20image/demo.gif" width="300" style="display:inline-block; margin-right:10px;">

---

<img src="https://github.com/user-attachments/assets/706cc52b-202c-4f99-ac9a-a260db88e88f" width="200" style="display:inline-block; margin-right:10px;">
<img src="https://github.com/user-attachments/assets/0633c47c-7988-4724-9237-c772a694c613" width="200" style="display:inline-block; margin-right:10px;">
