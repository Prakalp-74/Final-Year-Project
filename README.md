# Explainable AI for Brain Tumor Classification, Segmentation, and Automated Diagnostic Reporting Using Hybrid Deep Learning

Final Year B.E. Project (Artificial Intelligence)
Shri Chhatrapati Shivaji Maharaj College of Engineering (Ahmednagar Jilha Maratha Vidya Prasarak Samaj)

**Guide:** Prof. G. P. Joshi
**Team:** Omkar Joshi, Archit Bairagi, Prakalp Katla, Sarang Chaudhari

---

## Overview

Brain tumors need early and accurate diagnosis, but manual MRI interpretation is slow and varies between radiologists. Many deep learning models also behave as "black boxes", which limits clinical trust.

This project classifies brain MRI scans into four classes: **Glioma, Meningioma, Pituitary Tumor, No Tumor**. It uses a hybrid **ResNet-50 + Vision Transformer (ViT)** architecture fused through **Cross-Gated Multi-Path Attention Fusion (CG-MAF)**. **Grad-CAM** and **Integrated Gradients** explain every prediction.

## Key Features

- Hybrid ResNet-50 (local features) + ViT (global context) classifier
- CG-MAF fusion module trained with Gate-Consistency Loss
- Tumor segmentation (U-Net / Attention U-Net)
- Multi-modal fusion (MRI + CT + PET)
- Explainability with Grad-CAM and Integrated Gradients
- Automated PDF diagnostic report

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning | PyTorch, TensorFlow, Keras |
| Image processing | OpenCV, NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Explainability | Grad-CAM, Integrated Gradients |
| Environment | Jupyter Notebook, VS Code, Google Colab |

## Datasets

- Kaggle Brain Tumor MRI Dataset (classification): [text](https://www.kaggle.com/datasets/tombackert/brain-tumor-mri-data)
- BraTS (segmentation and multi-modal fusion)
- Mendeley Data: [text](https://data.mendeley.com/datasets/4xs94489rw/1)

Datasets are not stored in this repo. Download them and place them in `data/`.

## Project Structure

```
Final-Year-Project/
├── data/            # datasets (not tracked)
├── notebooks/       # experiments and EDA
├── src/             # models, training, XAI, reporting code
├── outputs/         # plots, saved models, PDF reports
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/Omkar-codes0105/Final-Year-Project.git
cd Final-Year-Project
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
```

## Methodology

1. Data collection
2. Preprocessing (resize, normalize, augment, split)
3. Feature extraction (ResNet-50 and ViT)
4. Feature fusion (CG-MAF)
5. Training (Gate-Consistency Loss)
6. Evaluation
7. Explainability (Grad-CAM, Integrated Gradients)
8. PDF report generation

## Evaluation Metrics

Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, Dice Coefficient, IoU

## Results

_To be added after model training._

## Future Scope

Cloud deployment, mobile healthcare apps, and real-time clinical integration with PACS/EHR systems.


