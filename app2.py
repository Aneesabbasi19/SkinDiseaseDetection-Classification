"""
Skin Disease Detection System
Classification First (Ensemble: ConvNeXtV2 + EfficientNet) + Detection for Visualization (YOLO)

Flow:
1. Ensemble Model (ConvNeXtV2 + EfficientNet) classifies the FULL image → Disease prediction
2. YOLO detects lesion location → Bounding box (visualization only)

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import timm
from ultralytics import YOLO
import os
import io
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import KeepTogether


# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="Skin Disease Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# CUSTOM CSS
# =====================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { padding: 1.5rem 2rem; background: #f8fafc; }
    
    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    .hero-badges {
        margin-top: 1rem;
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .hero-badge {
        background: rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 0.3rem 0.9rem;
        border-radius: 50px;
        font-size: 0.82rem;
        border: 1px solid rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
    }

    /* Result Cards */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    
    .malignant-card {
        background: linear-gradient(135deg, #ff416c 0%, #c0392b 100%);
    }
    
    .benign-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #1a1a1a !important;
    }

    .benign-card h2, .benign-card h3, .benign-card p {
        color: #1a1a1a !important;
    }

    .normal-card {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        color: #ffffff !important;
    }

    .normal-card h2, .normal-card h3, .normal-card p {
        color: #ffffff !important;
    }
    
    .inflammatory-card {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: #1a1a1a !important;
    }
    
    .inflammatory-card h2, .inflammatory-card h3, .inflammatory-card p {
        color: #1a1a1a !important;
    }

    .result-card h2 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.5rem 0; }
    .result-card h3 { font-size: 1.2rem; font-weight: 500; margin: 0; opacity: 0.9; }
    .result-card p  { font-size: 0.95rem; margin: 0.4rem 0 0 0; opacity: 0.8; }

    /* Normal skin banner */
    .normal-skin-banner {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        color: white;
    }
    .normal-skin-banner h2 { font-size: 2rem; font-weight: 700; margin: 0 0 0.4rem 0; color: white; }
    .normal-skin-banner h3 { font-size: 1.1rem; font-weight: 400; margin: 0; color: rgba(255,255,255,0.9); }
    .normal-skin-banner p  { font-size: 0.9rem; margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.75); }

    /* Stat Cards */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }

    .stat-value { font-size: 1.6rem; font-weight: 700; color: #1e293b; }
    .stat-label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Badges */
    .badge {
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .badge-malignant { background: linear-gradient(90deg,#ff416c,#c0392b); color: white; }
    .badge-benign    { background: linear-gradient(90deg,#11998e,#38ef7d); color: #1a1a1a; }
    .badge-inflammatory { background: linear-gradient(90deg,#f7971e,#ffd200); color: #1a1a1a; }

    /* Probability Bars */
    .prob-container { margin: 0.4rem 0; }
    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-size: 0.88rem;
        color: #374151;
        font-weight: 500;
    }
    .prob-bar-bg {
        background: #e5e7eb;
        border-radius: 8px;
        height: 20px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    /* Sidebar */
    .sidebar-section {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
        border: 1px solid #e2e8f0;
        font-size: 0.88rem;
    }
    
    .sidebar-section b { color: #1e293b; }

    /* Pipeline Steps */
    .pipeline-step {
        background: white;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.4rem 0;
        font-size: 0.88rem;
        color: #374151;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: white;
        padding: 4px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }

    /* Warning alert */
    .medical-warning {
        background: linear-gradient(135deg, #fff5f5, #ffe0e0);
        border: 2px solid #fc8181;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #742a2a;
    }

    /* Healthy notice */
    .healthy-notice {
        background: linear-gradient(135deg, #f0fff4, #c6f6d5);
        border: 2px solid #68d391;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #1a4731;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #94a3b8;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }

    /* Section divider */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================

class ConvNeXtV2Classifier(nn.Module):
    """Ensemble: ConvNeXtV2 + EfficientNet for 8-class skin disease classification"""
    
    def __init__(
        self, 
        model_name: str = 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        num_classes: int = 8, 
        dropout: float = 0.5,
        drop_path_rate: float = 0.2
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=False,
            drop_path_rate=drop_path_rate
        )
        
        if hasattr(self.backbone.head, 'fc'):
            in_features = self.backbone.head.fc.in_features
        else:
            in_features = self.backbone.head.in_features
        
        self.backbone.head.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =====================================================================
# CLASSIFICATION-FIRST PIPELINE
# =====================================================================

class SkinLesionPipeline:
    """
    Classification-First Pipeline
    
    Flow:
    1. Ensemble (ConvNeXtV2 + EfficientNet) classifies FULL image → Disease prediction
    2. YOLO detects lesion → Bounding box (visualization only)
    """
    
    MALIGNANT    = ['Melanoma', 'Basal cell carcinoma']
    BENIGN       = ['Melanocytic nevus', 'Vascular lesion', 'normal skin', 'Molluscum-Contagiosum']
    INFLAMMATORY = ['Psoriasis', 'acne']
    
    COLORS = {
        'malignant':    (255, 65, 108),
        'benign':       (17, 153, 142),
        'inflammatory': (247, 151, 30),
        'default':      (59, 130, 246)
    }
    
    def __init__(self, yolo_path: str, convnext_path: str, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.yolo_model = YOLO(yolo_path)
        
        checkpoint = torch.load(convnext_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        
        self.class_names = config.get('class_names', [
            'Basal cell carcinoma', 'Melanocytic nevus', 'Melanoma',
            'Molluscum-Contagiosum', 'Psoriasis', 'Vascular lesion', 'acne', 'normal skin'
        ])
        self.num_classes = len(self.class_names)
        self.img_size    = config.get('img_size', 224)
        self.mean        = config.get('mean', [0.485, 0.456, 0.406])
        self.std         = config.get('std',  [0.229, 0.224, 0.225])
        
        model_name = config.get('model_name', 'convnextv2_tiny.fcmae_ft_in22k_in1k')
        self.classifier = ConvNeXtV2Classifier(model_name=model_name, num_classes=self.num_classes)
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.tta_transforms = self._get_tta_transforms()
    
    def _get_tta_transforms(self):
        base = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ]
        return [
            transforms.Compose(base),
            transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                 transforms.RandomHorizontalFlip(p=1.0),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=self.mean, std=self.std)]),
            transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                 transforms.RandomVerticalFlip(p=1.0),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=self.mean, std=self.std)]),
            transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                 transforms.RandomRotation(degrees=(90, 90)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=self.mean, std=self.std)]),
            transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                 transforms.RandomRotation(degrees=(-90, -90)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=self.mean, std=self.std)]),
        ]
    
    def get_category(self, disease: str) -> str:
        if disease in self.MALIGNANT:    return 'malignant'
        if disease in self.BENIGN:       return 'benign'
        if disease in self.INFLAMMATORY: return 'inflammatory'
        return 'default'
    
    def get_color(self, disease: str) -> Tuple[int, int, int]:
        return self.COLORS.get(self.get_category(disease), self.COLORS['default'])
    
    def classify(self, image: Image.Image, use_tta: bool = True) -> Dict:
        self.classifier.eval()
        with torch.inference_mode():
            if use_tta:
                all_probs = []
                for t in self.tta_transforms:
                    img_tensor = t(image).unsqueeze(0).to(self.device)
                    outputs    = self.classifier(img_tensor)
                    probs      = F.softmax(outputs, dim=-1)
                    all_probs.append(probs)
                avg_probs = torch.stack(all_probs).mean(dim=0).squeeze()
            else:
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                outputs    = self.classifier(img_tensor)
                avg_probs  = F.softmax(outputs, dim=-1).squeeze()
        
        if avg_probs.dim() == 0:
            avg_probs = avg_probs.unsqueeze(0)
        
        confidence, predicted_idx = torch.max(avg_probs, dim=0)
        predicted_class = self.class_names[predicted_idx.item()]
        
        all_probabilities = {self.class_names[i]: float(avg_probs[i]) for i in range(self.num_classes)}
        sorted_probs      = dict(sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'category':   self.get_category(predicted_class),
            'probabilities': sorted_probs,
            'top3': list(sorted_probs.items())[:3]
        }
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        results    = self.yolo_model(image, conf=conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                })
        return detections
    
    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        use_tta: bool = True,
        show_detection: bool = True
    ) -> Tuple[np.ndarray, Dict, List[Dict]]:
        pil_image      = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        classification = self.classify(pil_image, use_tta)
        
        detections = []
        annotated  = image.copy()
        
        # Skip YOLO detection for normal skin — nothing to localise
        if show_detection and classification['predicted_class'] != 'normal skin':
            detections = self.detect(image, conf_threshold)
            color      = self.get_color(classification['predicted_class'])
            color_bgr  = (color[2], color[1], color[0])
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3)
                
                label = f"{classification['predicted_class']} ({classification['confidence']:.0%})"
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.65
                thickness  = 2
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 12, y1), color_bgr, -1)
                cv2.putText(annotated, label, (x1 + 6, y1 - 6), font, font_scale, (255, 255, 255), thickness)
        
        return annotated, classification, detections
    
    def classify_only(self, image: np.ndarray, use_tta: bool = True) -> Dict:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return self.classify(pil_image, use_tta)


# =====================================================================
# PDF REPORT GENERATOR
# =====================================================================

def generate_pdf_report(
    original_image: np.ndarray,
    annotated_image: np.ndarray,
    classification: Dict,
    detections: List[Dict],
    elapsed: float
) -> bytes:
    """Generate a beautiful PDF report with annotated image + predictions."""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8*cm,
        leftMargin=1.8*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )
    
    # Color palette
    CLR_DARK     = colors.HexColor('#0f2027')
    CLR_BLUE     = colors.HexColor('#2c5364')
    CLR_TEAL     = colors.HexColor('#11998e')
    CLR_RED      = colors.HexColor('#ff416c')
    CLR_ORANGE   = colors.HexColor('#f7971e')
    CLR_NORMAL   = colors.HexColor('#185a9d')
    CLR_LIGHT_BG = colors.HexColor('#f8fafc')
    CLR_BORDER   = colors.HexColor('#e2e8f0')
    CLR_TEXT     = colors.HexColor('#1e293b')
    CLR_MUTED    = colors.HexColor('#64748b')

    is_normal = classification['predicted_class'] == 'normal skin'

    CAT_COLORS = {
        'malignant':    CLR_RED,
        'benign':       CLR_TEAL,
        'inflammatory': CLR_ORANGE
    }
    
    cat_color = CLR_NORMAL if is_normal else CAT_COLORS.get(classification['category'], CLR_BLUE)
    
    # ---- Styles ----
    styles = getSampleStyleSheet()
    
    style_title = ParagraphStyle(
        'HeroTitle', fontName='Helvetica-Bold', fontSize=22,
        textColor=colors.white, alignment=TA_CENTER, spaceAfter=4
    )
    style_subtitle = ParagraphStyle(
        'Subtitle', fontName='Helvetica', fontSize=10,
        textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER, spaceAfter=2
    )
    style_section = ParagraphStyle(
        'Section', fontName='Helvetica-Bold', fontSize=12,
        textColor=CLR_DARK, spaceBefore=12, spaceAfter=6,
        borderPadding=(0, 0, 4, 0)
    )
    style_body = ParagraphStyle(
        'Body', fontName='Helvetica', fontSize=9,
        textColor=CLR_TEXT, leading=14
    )
    style_small = ParagraphStyle(
        'Small', fontName='Helvetica', fontSize=8,
        textColor=CLR_MUTED, leading=12
    )
    style_label = ParagraphStyle(
        'Label', fontName='Helvetica-Bold', fontSize=9,
        textColor=CLR_TEXT
    )
    style_disclaimer = ParagraphStyle(
        'Disclaimer', fontName='Helvetica-Oblique', fontSize=8,
        textColor=CLR_MUTED, alignment=TA_CENTER, leading=12
    )
    
    story = []
    
    # ---- HEADER BANNER ----
    header_data = [[
        Paragraph('🔬 Skin Disease Detection System', style_title),
    ]]
    header_sub = [[
        Paragraph('AI-Powered Analysis Report | Ensemble Model: ConvNeXtV2 + EfficientNet', style_subtitle),
    ]]
    
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), CLR_DARK),
        ('ROWPADDING',  (0,0), (-1,-1), 14),
        ('TOPPADDING',  (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('ROUNDEDCORNERS', [12]),
        ('BOX', (0,0), (-1,-1), 0, CLR_DARK),
    ]))
    story.append(header_table)
    
    sub_table = Table(header_sub, colWidths=[17*cm])
    sub_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CLR_BLUE),
        ('ROWPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 0, CLR_BLUE),
    ]))
    story.append(sub_table)
    story.append(Spacer(1, 0.3*cm))
    
    # ---- REPORT META ----
    now = datetime.now()
    meta_data = [
        [Paragraph('<b>Report Date:</b>', style_label), Paragraph(now.strftime('%B %d, %Y'), style_body),
         Paragraph('<b>Report Time:</b>', style_label), Paragraph(now.strftime('%I:%M %p'), style_body)],
        [Paragraph('<b>Analysis Model:</b>', style_label), Paragraph('Ensemble (ConvNeXtV2 + EfficientNet)', style_body),
         Paragraph('<b>Detection Model:</b>', style_label), Paragraph('YOLO v8', style_body)],
        [Paragraph('<b>Processing Time:</b>', style_label), Paragraph(f'{elapsed:.2f} seconds', style_body),
         Paragraph('<b>TTA Applied:</b>', style_label), Paragraph('Yes', style_body)],
    ]
    meta_table = Table(meta_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), CLR_LIGHT_BG),
        ('ROWPADDING',  (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('BOX',  (0,0), (-1,-1), 1, CLR_BORDER),
        ('GRID', (0,0), (-1,-1), 0.5, CLR_BORDER),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4*cm))
    
    # ---- IMAGES ----
    story.append(Paragraph('Image Analysis', style_section))
    story.append(HRFlowable(width='100%', thickness=1, color=CLR_BORDER))
    story.append(Spacer(1, 0.2*cm))
    
    def np_to_rl_image(np_img_bgr, max_width, max_height):
        """Convert numpy BGR image to ReportLab Image object."""
        rgb = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format='PNG')
        buf.seek(0)
        
        orig_w, orig_h = pil.size
        ratio = min(max_width / orig_w, max_height / orig_h)
        w = orig_w * ratio
        h = orig_h * ratio
        return RLImage(buf, width=w, height=h)
    
    orig_rl  = np_to_rl_image(original_image,  7.5*cm, 6.5*cm)
    annot_rl = np_to_rl_image(annotated_image, 7.5*cm, 6.5*cm)
    
    style_img_label = ParagraphStyle(
        'ImgLabel', fontName='Helvetica-Bold', fontSize=9,
        textColor=CLR_DARK, alignment=TA_CENTER, spaceAfter=4
    )
    
    annot_label = 'Original Image (No Lesion — Healthy Skin)' if is_normal else 'Annotated Image (YOLO Detection)'

    img_table = Table(
        [[Paragraph('Original Image', style_img_label), Paragraph(annot_label, style_img_label)],
         [orig_rl, annot_rl]],
        colWidths=[8.5*cm, 8.5*cm]
    )
    img_table.setStyle(TableStyle([
        ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',  (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
        ('BOX',   (0,0), (-1,-1), 1, CLR_BORDER),
        ('GRID',  (0,0), (-1,-1), 0.5, CLR_BORDER),
        ('ROWPADDING',  (0,0), (-1,-1), 8),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.5*cm))
    
    # ---- MAIN PREDICTION ----
    story.append(Paragraph('Classification Result', style_section))
    story.append(HRFlowable(width='100%', thickness=1, color=CLR_BORDER))
    story.append(Spacer(1, 0.2*cm))
    
    if is_normal:
        cat_emoji    = '✓'
        display_name = 'Normal Skin — No Disease Found'
        cat_label    = 'HEALTHY'
    else:
        cat_emoji    = {'malignant': '⚠', 'benign': '✓', 'inflammatory': '◆'}.get(classification['category'], '?')
        display_name = classification['predicted_class']
        cat_label    = classification['category'].upper()
    
    pred_style = ParagraphStyle(
        'Pred', fontName='Helvetica-Bold', fontSize=18,
        textColor=colors.white, alignment=TA_CENTER
    )
    conf_style = ParagraphStyle(
        'Conf', fontName='Helvetica', fontSize=12,
        textColor=colors.white, alignment=TA_CENTER
    )
    cat_style = ParagraphStyle(
        'Cat', fontName='Helvetica-Bold', fontSize=10,
        textColor=colors.white, alignment=TA_CENTER
    )
    
    pred_table = Table(
        [[Paragraph(f'{cat_emoji}  {display_name}', pred_style)],
         [Paragraph(f'Confidence: {classification["confidence"]:.1%}', conf_style)],
         [Paragraph(f'Category: {cat_label}', cat_style)]],
        colWidths=[17*cm]
    )
    pred_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), cat_color),
        ('ROWPADDING',    (0,0), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (0,0),   16),
        ('BOTTOMPADDING', (0,2), (0,2),   16),
        ('ROUNDEDCORNERS', [10]),
        ('BOX', (0,0), (-1,-1), 0, cat_color),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.5*cm))
    
    # ---- PROBABILITY DISTRIBUTION ----
    story.append(Paragraph('Probability Distribution', style_section))
    story.append(HRFlowable(width='100%', thickness=1, color=CLR_BORDER))
    story.append(Spacer(1, 0.2*cm))
    
    prob_rows = [
        [Paragraph('<b>Disease</b>', style_label),
         Paragraph('<b>Category</b>', style_label),
         Paragraph('<b>Probability</b>', style_label)]
    ]
    
    pipeline_cat_map = {
        'Melanoma': 'Malignant', 'Basal cell carcinoma': 'Malignant',
        'Melanocytic nevus': 'Benign', 'Vascular lesion': 'Benign',
        'normal skin': 'Healthy', 'Molluscum-Contagiosum': 'Benign',
        'Psoriasis': 'Inflammatory', 'acne': 'Inflammatory'
    }
    
    for disease, prob in classification['probabilities'].items():
        cat_str = pipeline_cat_map.get(disease, 'Other')
        cat_c   = {
            'Malignant': CLR_RED,
            'Benign': CLR_TEAL,
            'Inflammatory': CLR_ORANGE,
            'Healthy': CLR_NORMAL
        }.get(cat_str, CLR_BLUE)
        is_pred = disease == classification['predicted_class']
        
        display_disease = 'Normal Skin (No Disease)' if disease == 'normal skin' else disease

        disease_para = Paragraph(
            f'<b>{display_disease}</b>' if is_pred else display_disease,
            ParagraphStyle('dp', fontName='Helvetica-Bold' if is_pred else 'Helvetica',
                           fontSize=9, textColor=CLR_TEXT)
        )
        cat_para = Paragraph(
            cat_str,
            ParagraphStyle('cp', fontName='Helvetica-Bold', fontSize=8, textColor=cat_c)
        )
        prob_para = Paragraph(
            f'<b>{prob:.1%}</b>' if is_pred else f'{prob:.1%}',
            ParagraphStyle('pp', fontName='Helvetica-Bold' if is_pred else 'Helvetica',
                           fontSize=9, textColor=CLR_TEXT, alignment=TA_RIGHT)
        )
        prob_rows.append([disease_para, cat_para, prob_para])
    
    prob_table = Table(prob_rows, colWidths=[8*cm, 4*cm, 5*cm])
    
    table_style = [
        ('BACKGROUND',  (0,0), (-1,0),  CLR_DARK),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('ROWPADDING',  (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('BOX',   (0,0), (-1,-1), 1, CLR_BORDER),
        ('GRID',  (0,0), (-1,-1), 0.5, CLR_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CLR_LIGHT_BG]),
        ('ROUNDEDCORNERS', [8]),
    ]
    
    # Highlight predicted row
    for i, (disease, _) in enumerate(classification['probabilities'].items(), start=1):
        if disease == classification['predicted_class']:
            table_style.append(('BACKGROUND', (0,i), (-1,i), colors.HexColor('#dbeafe')))
            break
    
    prob_table.setStyle(TableStyle(table_style))
    story.append(prob_table)
    story.append(Spacer(1, 0.5*cm))
    
    # ---- DETECTION SUMMARY ----
    if detections:
        story.append(Paragraph('Lesion Detection Summary (YOLO)', style_section))
        story.append(HRFlowable(width='100%', thickness=1, color=CLR_BORDER))
        story.append(Spacer(1, 0.2*cm))
        
        det_rows = [[
            Paragraph('<b>#</b>', style_label),
            Paragraph('<b>Bounding Box (x1, y1, x2, y2)</b>', style_label),
            Paragraph('<b>Detection Confidence</b>', style_label)
        ]]
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det['bbox']
            det_rows.append([
                Paragraph(str(i), style_body),
                Paragraph(f'({x1}, {y1}, {x2}, {y2})', style_body),
                Paragraph(f'{det["confidence"]:.1%}', style_body)
            ])
        
        det_table = Table(det_rows, colWidths=[2*cm, 9*cm, 6*cm])
        det_table.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,0),  CLR_BLUE),
            ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
            ('ROWPADDING',  (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('BOX',   (0,0), (-1,-1), 1, CLR_BORDER),
            ('GRID',  (0,0), (-1,-1), 0.5, CLR_BORDER),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CLR_LIGHT_BG]),
        ]))
        story.append(det_table)
        story.append(Spacer(1, 0.5*cm))

    # ---- NORMAL SKIN NOTE ----
    if is_normal:
        healthy_data = [[Paragraph(
            f'<b>✓ HEALTHY SKIN: No Disease Detected</b><br/>'
            f'The model classified this image as <b>Normal Skin</b> '
            f'with a confidence of <b>{classification["confidence"]:.1%}</b>. '
            f'No pathological lesion was identified. Regular skin check-ups are still recommended '
            f'to monitor any future changes.',
            ParagraphStyle('Healthy', fontName='Helvetica', fontSize=9,
                           textColor=colors.HexColor('#1a4731'), leading=14)
        )]]
        healthy_table = Table(healthy_data, colWidths=[17*cm])
        healthy_table.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,-1), colors.HexColor('#f0fff4')),
            ('BOX',         (0,0), (-1,-1), 1.5, colors.HexColor('#68d391')),
            ('ROWPADDING',  (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('ROUNDEDCORNERS', [8]),
        ]))
        story.append(healthy_table)
        story.append(Spacer(1, 0.4*cm))
    
    # ---- CLINICAL NOTE (malignant only) ----
    elif classification['category'] == 'malignant':
        warn_data = [[Paragraph(
            f'<b>⚠ CLINICAL ALERT: Potential Malignant Lesion Detected</b><br/>'
            f'The model has classified this lesion as <b>{classification["predicted_class"]}</b> '
            f'with a confidence of <b>{classification["confidence"]:.1%}</b>. '
            f'Please consult a board-certified dermatologist immediately for a definitive diagnosis and appropriate treatment plan.',
            ParagraphStyle('Warn', fontName='Helvetica', fontSize=9,
                           textColor=colors.HexColor('#742a2a'), leading=14)
        )]]
        warn_table = Table(warn_data, colWidths=[17*cm])
        warn_table.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,-1), colors.HexColor('#fff5f5')),
            ('BOX',         (0,0), (-1,-1), 1.5, CLR_RED),
            ('ROWPADDING',  (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('ROUNDEDCORNERS', [8]),
        ]))
        story.append(warn_table)
        story.append(Spacer(1, 0.4*cm))
    
    # ---- DISCLAIMER ----
    story.append(HRFlowable(width='100%', thickness=0.5, color=CLR_BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        'DISCLAIMER: This report is generated by an AI system for educational and research purposes only. '
        'It does not constitute medical advice and should not be used as a substitute for professional medical consultation. '
        'Always consult a qualified healthcare provider for diagnosis and treatment decisions.',
        style_disclaimer
    ))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        f'Generated by Skin Disease Detection System | {now.strftime("%Y-%m-%d %H:%M:%S")}',
        style_disclaimer
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =====================================================================
# CACHING
# =====================================================================

@st.cache_resource
def load_pipeline(yolo_path: str, convnext_path: str):
    return SkinLesionPipeline(yolo_path, convnext_path)


# =====================================================================
# HELPERS
# =====================================================================

def get_result_card_class(category: str, predicted_class: str = '') -> str:
    """Return CSS class for result card. Normal skin gets its own distinctive card."""
    if predicted_class == 'normal skin':
        return 'normal-card'
    return {
        'malignant':    'malignant-card',
        'benign':       'benign-card',
        'inflammatory': 'inflammatory-card'
    }.get(category, '')


def get_display_name(predicted_class: str) -> str:
    """Human-friendly label for the predicted class."""
    if predicted_class == 'normal skin':
        return 'Normal Skin — No Disease Found'
    return predicted_class


def get_cat_label(category: str, predicted_class: str) -> str:
    """Human-friendly category label."""
    if predicted_class == 'normal skin':
        return 'HEALTHY'
    return category.upper()


def get_cat_emoji(category: str, predicted_class: str) -> str:
    if predicted_class == 'normal skin':
        return '✅'
    return {'malignant': '⚠️', 'benign': '✅', 'inflammatory': '🔶'}.get(category, '❓')


def render_result_card(classification: Dict, extra_info: str = '') -> None:
    """Render the main result card, with special handling for normal skin."""
    predicted = classification['predicted_class']
    is_normal = predicted == 'normal skin'

    card_class   = get_result_card_class(classification['category'], predicted)
    display_name = get_display_name(predicted)
    cat_label    = get_cat_label(classification['category'], predicted)
    emoji        = get_cat_emoji(classification['category'], predicted)

    st.markdown(f"""
    <div class="result-card {card_class}">
        <h2>{emoji} {display_name}</h2>
        <h3>Confidence: {classification['confidence']:.1%}</h3>
        <p>Category: {cat_label}{' &nbsp;|&nbsp; ' + extra_info if extra_info else ''}</p>
    </div>
    """, unsafe_allow_html=True)


def render_clinical_notice(classification: Dict) -> None:
    """Show appropriate clinical notice based on prediction."""
    predicted = classification['predicted_class']
    is_normal = predicted == 'normal skin'

    if is_normal:
        st.markdown("""
        <div class="healthy-notice">
            <b>✅ No Disease Detected — Skin Appears Healthy</b><br>
            The AI model found no signs of a skin condition in this image. 
            Regular annual skin check-ups with a dermatologist are still recommended, 
            especially if you notice any new or changing spots.
        </div>
        """, unsafe_allow_html=True)
    elif classification['category'] == 'malignant':
        st.markdown(f"""
        <div class="medical-warning">
            <b>⚠️ Potential Malignant Lesion: {predicted}</b><br>
            Confidence: {classification['confidence']:.1%}<br><br>
            Please consult a board-certified dermatologist immediately for a professional diagnosis.
        </div>
        """, unsafe_allow_html=True)


def create_probability_bar(disease: str, prob: float, category: str, is_normal_skin: bool = False) -> str:
    colors_map = {
        'malignant':    '#ff416c',
        'benign':       '#11998e',
        'inflammatory': '#f7971e'
    }
    # Normal skin gets a distinct teal-blue
    if is_normal_skin:
        color = '#185a9d'
    else:
        color = colors_map.get(category, '#3b82f6')

    display = 'Normal Skin (No Disease)' if disease == 'normal skin' else disease
    width = max(int(prob * 100), 1)
    return f"""
    <div class="prob-container">
        <div class="prob-label"><span>{display}</span><span>{prob:.1%}</span></div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="background: {color}; width: {width}%;"></div>
        </div>
    </div>
    """


def image_to_png_bytes(image: np.ndarray) -> bytes:
    _, buffer = cv2.imencode(".png", image)
    return buffer.tobytes()


# =====================================================================
# SIDEBAR
# =====================================================================

def render_sidebar():
    st.sidebar.markdown("## ⚙️ Configuration")
    
    st.sidebar.markdown("### 📂 Model Paths")
    yolo_path = st.sidebar.text_input(
        "YOLO Model (.pt)",
        value="C:/Users/talha/OneDrive/Desktop/Fyp/best.pt",
        help="Path to YOLO v8 detection model"
    )
    convnext_path = st.sidebar.text_input(
        "Ensemble Model (.pth)",
        value="C:/Users/talha/OneDrive/Desktop/Fyp/best_model.pth",
        help="Path to ConvNeXtV2 + EfficientNet ensemble model"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Inference Settings")
    
    conf_threshold = st.sidebar.slider(
        "Detection Confidence", 0.1, 0.9, 0.25, 0.05,
        help="YOLO confidence threshold for lesion detection"
    )
    use_tta = st.sidebar.checkbox(
        "Test Time Augmentation (TTA)", value=True,
        help="Ensemble predictions over augmented views"
    )
    show_detection = st.sidebar.checkbox(
        "Show YOLO Bounding Boxes", value=True,
        help="Overlay detection boxes on the result image (skipped for Normal Skin)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Pipeline")
    st.sidebar.markdown("""
    <div class="pipeline-step">
        <b>Step 1 — Classification</b><br>
        Ensemble (ConvNeXtV2 + EfficientNet) analyses the full image and outputs the disease class.
    </div>
    <div class="pipeline-step" style="border-color:#11998e;">
        <b>Step 2 — Detection</b><br>
        YOLO localises the lesion and draws a bounding box (skipped for Normal Skin).
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Disease Classes")
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <b style="color:#ff416c;">🔴 Malignant</b><br>
        Melanoma · Basal Cell Carcinoma<br><br>
        <b style="color:#11998e;">🟢 Benign</b><br>
        Melanocytic Nevus · Vascular Lesion<br>Molluscum-Contagiosum<br><br>
        <b style="color:#185a9d;">🔵 Healthy</b><br>
        Normal Skin — No Disease Found<br><br>
        <b style="color:#f7971e;">🟠 Inflammatory</b><br>
        Psoriasis · Acne
    </div>
    """, unsafe_allow_html=True)
    
    device_icon = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.sidebar.markdown(f"**Device:** {device_icon}")
    
    return yolo_path, convnext_path, conf_threshold, use_tta, show_detection


# =====================================================================
# MAIN
# =====================================================================

def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">🔬 Skin Disease Detection System</div>
        <div class="hero-subtitle">AI-Powered Lesion Classification + Detection</div>
        <div class="hero-badges">
            <span class="hero-badge">🧠 Ensemble Model</span>
            <span class="hero-badge">🎯 YOLO Detection</span>
            <span class="hero-badge">📊 8 Disease Classes</span>
            <span class="hero-badge">⚡ TTA Support</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    yolo_path, convnext_path, conf_threshold, use_tta, show_detection = render_sidebar()
    
    models_exist = os.path.exists(yolo_path) and os.path.exists(convnext_path)
    
    if not models_exist:
        st.warning("⚠️ Please update the model paths in the sidebar to get started.")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**YOLO Model** (.pt)\nUsed for lesion localisation & bounding box visualisation.")
        with c2:
            st.info("**Ensemble Model** (.pth)\nConvNeXtV2 + EfficientNet for disease classification.")
        return
    
    try:
        with st.spinner("🔄 Loading models…"):
            pipeline = load_pipeline(yolo_path, convnext_path)
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.exception(e)
        return
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyse", "📸 Camera Capture", "🔬 Classify Only", "ℹ️ About"])
    
    # =====================================================================
    # TAB 1 — UPLOAD
    # =====================================================================
    with tab1:
        st.markdown("### 📤 Upload Skin Lesion Image")
        st.caption("The ensemble model classifies the full image; YOLO localises the lesion for visualisation (skipped for Normal Skin).")
        
        uploaded_file = st.file_uploader(
            "Choose an image…",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear, well-lit image of the skin area",
            key="upload_full"
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📷 Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if st.button("🔍 Analyse Image", type="primary", use_container_width=True, key="analyze_full"):
                with st.spinner("🔬 Running ensemble classification & YOLO detection…"):
                    start_time = time.time()
                    annotated, classification, detections = pipeline.predict(
                        image, conf_threshold, use_tta, show_detection
                    )
                    elapsed = time.time() - start_time
                
                is_normal = classification['predicted_class'] == 'normal skin'

                with col2:
                    st.markdown("#### 🎯 Annotated Result")
                    if is_normal:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.caption("ℹ️ No bounding box — healthy skin, no lesion to localise.")
                    else:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🎯 Classification Result")
                
                render_result_card(classification, extra_info=f"Category: {get_cat_label(classification['category'], classification['predicted_class'])}")
                
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    first_word = 'Normal' if is_normal else classification['predicted_class'].split()[0]
                    st.markdown(f"""<div class="stat-card"><div class="stat-value">{first_word}</div><div class="stat-label">Prediction</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="stat-card"><div class="stat-value">{classification['confidence']:.1%}</div><div class="stat-label">Confidence</div></div>""", unsafe_allow_html=True)
                with m3:
                    det_count = '—' if is_normal else (len(detections) if show_detection else '—')
                    st.markdown(f"""<div class="stat-card"><div class="stat-value">{det_count}</div><div class="stat-label">Lesions Found</div></div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""<div class="stat-card"><div class="stat-value">{elapsed:.2f}s</div><div class="stat-label">Process Time</div></div>""", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Probability distribution
                st.markdown("### 📊 Probability Distribution")
                for disease, prob in classification['probabilities'].items():
                    category   = pipeline.get_category(disease)
                    normal_bar = (disease == 'normal skin')
                    st.markdown(create_probability_bar(disease, prob, category, normal_bar), unsafe_allow_html=True)
                
                # Clinical notice
                render_clinical_notice(classification)
                
                st.markdown("---")
                
                # ── Download section ──
                st.markdown("### 💾 Download Results")
                dl1, dl2 = st.columns(2)
                
                # Use original image for download when normal skin
                download_img = image if is_normal else annotated

                with dl1:
                    st.download_button(
                        label="📸 Download Annotated Image",
                        data=image_to_png_bytes(download_img),
                        file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with dl2:
                    with st.spinner("Generating PDF report…"):
                        pdf_bytes = generate_pdf_report(
                            image, download_img, classification, detections, elapsed
                        )
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"skin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    
    # =====================================================================
    # TAB 2 — CAMERA
    # =====================================================================
    with tab2:
        st.markdown("### 📸 Capture from Camera")
        camera_image = st.camera_input("Take a picture", key="camera_full")
        
        if camera_image is not None:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if st.button("🔍 Analyse Captured Image", type="primary", use_container_width=True, key="analyze_camera"):
                with st.spinner("🔬 Analysing…"):
                    start_time = time.time()
                    annotated, classification, detections = pipeline.predict(
                        image, conf_threshold, use_tta, show_detection
                    )
                    elapsed = time.time() - start_time
                
                is_normal = classification['predicted_class'] == 'normal skin'
                download_img = image if is_normal else annotated

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 📷 Captured")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.markdown("#### 🎯 Result")
                    if is_normal:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.caption("ℹ️ No bounding box — healthy skin detected.")
                    else:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                render_result_card(classification, extra_info=f"{elapsed:.2f}s")
                render_clinical_notice(classification)
                
                # Probability bars
                st.markdown("### 📊 Probability Distribution")
                for disease, prob in classification['probabilities'].items():
                    category   = pipeline.get_category(disease)
                    normal_bar = (disease == 'normal skin')
                    st.markdown(create_probability_bar(disease, prob, category, normal_bar), unsafe_allow_html=True)
                
                # Download
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button(
                        "📸 Download Result Image",
                        data=image_to_png_bytes(download_img),
                        file_name=f"camera_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
                with dc2:
                    pdf_bytes = generate_pdf_report(image, download_img, classification, detections, elapsed)
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"skin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
    
    # =====================================================================
    # TAB 3 — CLASSIFY ONLY
    # =====================================================================
    with tab3:
        st.markdown("### 🔬 Classification Only (No Detection)")
        st.caption("Faster — skips YOLO detection and runs the ensemble model only.")
        
        uploaded_file_classify = st.file_uploader(
            "Choose an image…",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="upload_classify"
        )
        
        if uploaded_file_classify is not None:
            file_bytes = np.asarray(bytearray(uploaded_file_classify.read()), dtype=np.uint8)
            image      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📷 Input Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if st.button("🔬 Classify", type="primary", use_container_width=True, key="classify_only"):
                with st.spinner("🔬 Running ensemble classification…"):
                    start_time     = time.time()
                    classification = pipeline.classify_only(image, use_tta)
                    elapsed        = time.time() - start_time
                
                is_normal = classification['predicted_class'] == 'normal skin'

                with col2:
                    st.markdown("#### 📊 Probabilities")
                    for disease, prob in classification['probabilities'].items():
                        category   = pipeline.get_category(disease)
                        normal_bar = (disease == 'normal skin')
                        st.markdown(create_probability_bar(disease, prob, category, normal_bar), unsafe_allow_html=True)
                
                st.markdown("---")
                render_result_card(classification, extra_info=f"{elapsed:.2f}s")
                render_clinical_notice(classification)
                
                # PDF (no detection boxes)
                pdf_bytes = generate_pdf_report(image, image, classification, [], elapsed)
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"skin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    # =====================================================================
    # TAB 4 — ABOUT
    # =====================================================================
    with tab4:
        st.markdown("### ℹ️ About This System")
        
        st.markdown("""
        #### 🔄 Pipeline Architecture
        
        ```
        ┌────────────────────────────────────────────────────────────┐
        │                                                            │
        │  INPUT IMAGE                                               │
        │       │                                                    │
        │       ├──▶ Ensemble Model (ConvNeXtV2 + EfficientNet)     │
        │       │         └──▶ Disease Classification (8 classes)   │
        │       │                                                    │
        │       └──▶ YOLO v8  (skipped for Normal Skin)            │
        │                 └──▶ Lesion Localisation (bbox only)      │
        │                                                            │
        │  Classification is INDEPENDENT of detection               │
        └────────────────────────────────────────────────────────────┘
        ```
        
        #### 🧠 Model Details
        
        | Component | Architecture | Role |
        |-----------|-------------|------|
        | Ensemble Classifier | ConvNeXtV2 + EfficientNet | Predicts disease class from full image |
        | Detector | YOLO v8 | Localises lesion for visualisation (skipped for Normal Skin) |
        
        #### 📊 Disease Classes
        
        | Category | Diseases | Risk |
        |----------|----------|------|
        | 🔴 Malignant | Melanoma, Basal Cell Carcinoma | High |
        | 🟢 Benign | Nevus, Vascular, Molluscum | Low |
        | 🔵 Healthy | **Normal Skin — No Disease Found** | None |
        | 🟠 Inflammatory | Psoriasis, Acne | Medium |
        
        #### ⚠️ Disclaimer
        
        > **This system is for educational and research purposes only.**  
        > It does not constitute medical advice. Always consult a qualified dermatologist for diagnosis and treatment.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        🔬 Skin Disease Classification System &nbsp;|&nbsp; Ensemble: ConvNeXtV2 + EfficientNet &nbsp;|&nbsp; Detection: YOLO v8<br>
        ⚠️ For educational purposes only — not medical advice.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()