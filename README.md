# AI-Generated Synthetic Media Detection using Hybrid CNN–Vision Transformer

A deep learning framework for detecting **AI-generated synthetic images and videos** using a **Hybrid CNN–Vision Transformer–Attention architecture**. The system integrates **local texture extraction, global contextual reasoning, and explainable AI** to accurately classify images as **REAL or FAKE**.

📄 **Research Paper Accepted at IEEE Conference**

---

# Overview

The rapid growth of generative AI models has made it increasingly difficult to distinguish synthetic media from authentic content. This project proposes a **Hybrid CNN–Vision Transformer architecture** that combines the strengths of convolutional neural networks and transformer-based models for improved synthetic media detection.

The system performs:

- Image classification (REAL vs FAKE)
- Video deepfake detection through frame analysis
- Explainability using Grad-CAM
- Web-based deployment for real-time inference

---

# Key Features

• Hybrid **CNN + Vision Transformer + Attention** architecture  
• Detects **AI-generated images** with high accuracy  
• **Grad-CAM visualization** for explainable predictions  
• **Web interface** for image and video detection  
• **Frame-level analysis** for video deepfake detection  
• **Research publication accepted at IEEE**

---

# Model Architecture

The proposed architecture integrates three key components:

### 1. CNN Feature Extraction
Extracts **local spatial features** such as texture inconsistencies and pixel-level artifacts.

### 2. Vision Transformer (ViT)
Captures **global contextual relationships** across image patches using multi-head self-attention.

### 3. Attention Refinement Module
Enhances discriminative feature learning by focusing on important regions before classification.

Final classification is performed using a **fully connected layer with softmax activation**.

---

# Dataset

This project uses the **CIFAKE dataset**, which contains:

- **120,000 images**
- **60,000 REAL images**
- **60,000 AI-generated images**
- Image size: **32 × 32 RGB**

Dataset split:

- **Training:** 80%
- **Testing:** 20%

---

# Model Performance

| Metric | Score |
|------|------|
| Accuracy | **93%** |
| AUC Score | **0.98** |
| Precision | 0.93 |
| Recall | 0.93 |
| F1-Score | 0.93 |

The high **AUC score** indicates strong class separability between real and synthetic images.

---

# Evaluation Metrics

The model was evaluated using:

- Confusion Matrix
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC Score

---

# Explainable AI (Grad-CAM)

Grad-CAM was used to visualize **regions influencing model predictions**.

This helps verify that the model focuses on:

- texture artifacts
- background inconsistencies
- generative model distortions

---

# Web Deployment

A web-based interface was developed for real-time detection.

### Image Detection
Users can upload an image and receive:

- REAL / FAKE classification
- confidence score
- Grad-CAM heatmap

### Video Detection
The system performs:

1. Frame extraction
2. Frame-level classification
3. Majority voting
4. Final video classification

---

# Technologies Used

### Programming Language
Python

### Deep Learning Framework
PyTorch

### Libraries
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- TorchVision  

### Tools
- Visual Studio Code
- Git
- Web Deployment

---

# Project Structure
```
Synthetic-Media-Detection
│
├── dataset
│
├── models
│ ├── cnn_block.py
│ ├── vit_block.py
│ ├── hybrid_model.py
│
├── training
│ ├── train_model.py
│ ├── evaluation.py
│
├── web_app
│ ├── frontend
│ ├── backend
│
├── results
│ ├── confusion_matrix
│ ├── roc_curve
│ ├── gradcam_outputs
│
└── README.md
```


---

# Research Publication

**Title:**  
AI-Generated Synthetic Media Detection using Hybrid CNN–Transformer Architecture

**Conference:**  
Accepted at **IEEE International Conference**

---

# Applications

- Deepfake detection  
- Media authentication  
- Digital forensics  
- Social media content verification  
- AI security systems

---

# Future Improvements

- Extend model to **higher resolution datasets**
- Improve **video-level detection using temporal transformers**
- Enhance **adversarial robustness**
- Optimize for **real-time edge deployment**

---

# Author
Madhu Kanth Kodali
Final Year Project – Computer Science

---

# License

This project is for **research and educational purposes**.
