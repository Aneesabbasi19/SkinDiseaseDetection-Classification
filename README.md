<div align="center">

# Skin Disease Detection & Classification System

A dual-model deep learning ensemble for automated skin disease detection, developed as a Final Year Project in collaboration with the **National Centre for Physics (NCP), Islamabad**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](#)

</div>

---

## Overview

Skin disease often goes undiagnosed or misdiagnosed due to limited access to dermatologists, especially in underserved regions. This project addresses that gap with a two-stage computer vision pipeline: a detection model that locates a skin lesion in an image, and a classification ensemble that identifies which of 8 disease categories it belongs to — all wrapped in a simple web interface a non-technical user can operate.

## How It Works

```
Image Upload → YOLOv8m Lesion Detection → Dual-Model Classification → Diagnosis + Confidence Score
```

1. **Detection** — YOLOv8m locates the lesion in the uploaded image and draws a bounding box around it
2. **Classification** — The cropped region is passed through two independent architectures whose outputs are combined into a single ensemble prediction
3. **Output** — The app displays the predicted condition, a confidence score, and (optionally) a downloadable PDF report

## Model Architecture

| Component | Model | Purpose |
|---|---|---|
| Lesion Detection | YOLOv8m | Real-time localization of the affected region |
| Classifier 1 | ConvNeXtV2-Base | Fine-grained visual feature extraction |
| Classifier 2 | EfficientNetV2-M | Complementary classification signal |
| Ensemble | Weighted combination of both classifiers | Final diagnosis across 8 disease categories |

**Training techniques used:**
- Focal Loss (γ = 2.5) to counter class imbalance across disease categories
- Stochastic Weight Averaging (SWA) for better generalization
- Exponential Moving Average (EMA) for training stability

## Results

| Metric | Score |
|---|---|
| Ensemble classification accuracy | **91.42%** |
| Disease categories covered | 8 |
| Detection | Real-time, with bounding box visualization |

## Application

The model is served through a **Streamlit** web app with:
- Drag-and-drop image upload
- Bounding box visualization of the detected lesion
- Per-class confidence scores
- Downloadable PDF report of results

## Tech Stack

`Python` · `PyTorch` · `TorchVision` · `Timm` · `Ultralytics YOLOv8` · `Streamlit` · `OpenCV` · `ReportLab`

## Getting Started

### Prerequisites
- Python 3.10+
- ~500MB free disk space for model weights

### 1. Clone the repository
```bash
git clone https://github.com/Aneesabbasi19/SkinDiseaseDetection-Classification.git
cd SkinDiseaseDetection-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> For GPU acceleration, install PyTorch separately using the correct CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) before running the command above.

### 3. Download the model weights
Model files are hosted as GitHub Release assets (too large for the main repo):

| File | Size | Download |
|---|---|---|
| `best_model.pth` — Ensemble classifier weights | ~325 MB | [Download](https://github.com/Aneesabbasi19/SkinDiseaseDetection-Classification/releases/download/v1.0.0/best_model.pth) |
| `best.pt` — YOLOv8m detector weights | ~148 MB | [Download](https://github.com/Aneesabbasi19/SkinDiseaseDetection-Classification/releases/download/v1.0.0/best.pt) |

Place both files in the project root directory after downloading.

### 4. Run the app
```bash
streamlit run app2.py
```
Then open `http://localhost:8501` in your browser.

## Project Structure

```
SkinDiseaseDetection-Classification/
├── app2.py             # Streamlit application entry point
├── requirements.txt    # Python dependencies
└── README.md
```
*(Model weight files are downloaded separately — see step 3 above)*

## Author

**Anees Abbasi**
AI Engineer — Machine Learning · Computer Vision · NLP

[LinkedIn](#) · [Portfolio](#) · aneesabbasitg@gmail.com

---

<sub>Developed as a Final Year Project — BS Artificial Intelligence, Pak-Austria Fachhochschule Institute of Applied Science and Technology, in collaboration with the National Centre for Physics (NCP), Islamabad.</sub>
