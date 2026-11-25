# Alzheimer's Disease Classification Using Brain MRI

## Project Overview

This project is a deep-learning-based research prototype for binary classification of brain MRI images into:

- **Normal:** Non-Alzheimer's MRI
- **Alzheimer's:** MRI belonging to the Alzheimer's class

Five deep learning architectures were trained and evaluated:

- Custom CNN
- ResNet50
- DenseNet121
- EfficientNetB0
- Vision Transformer (ViT)

DenseNet121 was selected for the backend implementation based on its overall classification performance.

## Technology Stack

- Python
- PyTorch
- TorchVision
- FastAPI
- Pillow
- Uvicorn

## Model Information

- Selected architecture: DenseNet121
- Checkpoint: Seed 7
- Input image size: 224 × 224
- Input channels: RGB
- Number of classes: 2
- Class 0: Normal
- Class 1: Alzheimer's

## Project Structure

```text
alzheimer-mri-project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── best_model.pth
│   └── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd alzheimer-mri-project
```

### 2. Navigate to the backend

```bash
cd backend
```

### 3. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start the backend server

```bash
python -m uvicorn app.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Backend and model health check |
| POST | `/predict` | Predict the class of an uploaded MRI image |

## API Documentation

After starting the server, open:

`http://127.0.0.1:8000/docs`

The interactive Swagger UI allows users to upload MRI images and test predictions.

## Prediction Output

The prediction endpoint returns:

- Predicted class
- Class ID
- Model confidence score
- Research-use disclaimer

## Important Disclaimer

This project is an academic research prototype. Its predictions are experimental and are not intended to diagnose Alzheimer's disease or replace professional medical evaluation.

The dataset was divided at image level rather than patient level. Therefore, possible patient-level data leakage is a limitation when interpreting the reported evaluation results.