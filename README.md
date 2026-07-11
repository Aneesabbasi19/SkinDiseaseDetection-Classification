# 🩺 Skin Disease Detection & Classification System

A dual-model deep learning ensemble for automated skin disease detection and classification, developed as a Final Year Project in collaboration with the **National Centre for Physics (NCP), Islamabad**.

## 🎯 Overview

This system combines two state-of-the-art classification architectures with a real-time object detection model to identify and localize skin lesions, then classify them across **8 skin disease categories** with clinical-grade accuracy.

## 🧠 Model Architecture

- **Classification Ensemble:** ConvNeXtV2-Base + EfficientNetV2-M
- **Lesion Detection & Localization:** YOLOv8m (real-time bounding box detection)
- **Training Enhancements:**
  - Focal Loss (γ = 2.5) — to handle class imbalance across disease categories
  - Stochastic Weight Averaging (SWA)
  - Exponential Moving Average (EMA)

## 📊 Results

- **91.42% ensemble accuracy** across 8 skin disease categories
- Real-time lesion localization via YOLOv8m
- Deployed with confidence scores and bounding box visualization for interpretability

## 🖥️ Application

Built and deployed as an interactive **Streamlit web application** allowing:
- Image upload for instant analysis
- Bounding box visualization of detected lesions
- Confidence scores per prediction
- (Optional) PDF report generation for results

## 🛠️ Tech Stack

`Python` · `PyTorch` · `TorchVision` · `Timm` · `Ultralytics (YOLOv8)` · `Streamlit` · `ReportLab`

## 📁 Project Structure

```
skin-disease-detection-ensemble/
├── app2.py              # Streamlit application
├── requirements.txt      # Python dependencies
└── README.md

(best_model.pth and best.pt are hosted separately on Hugging Face Hub — see Model Weights section below)
```

## ⚙️ Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/Aneesabbasi19/skin-disease-detection-ensemble.git
cd skin-disease-detection-ensemble
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app2.py
```

4. Open your browser at `http://localhost:8501`

## 📌 Model Weights

Due to file size, trained model weights are hosted as GitHub Release assets rather than in the main repository:

- **Ensemble classifier** (`best_model.pth`, ~325MB): [Download](https://github.com/Aneesabbasi19/skin-disease-detection-ensemble/releases/download/v1.0.0/best_model.pth)
- **YOLOv8m detector** (`best.pt`, ~148MB): [Download](https://github.com/Aneesabbasi19/skin-disease-detection-ensemble/releases/download/v1.0.0/best.pt)

After downloading, place both files in the project root directory before running the app.

## 👤 Author

**Anees Abbasi**
AI Engineer | Machine Learning · Computer Vision · NLP
[LinkedIn](#) · [Portfolio](#) · aneesabbasitg@gmail.com

---
*Developed as a Final Year Project — BS Artificial Intelligence, Pak-Austria Fachhochschule Institute of Applied Science and Technology, in collaboration with the National Centre for Physics (NCP), Islamabad.*
