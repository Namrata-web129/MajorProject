# Self-Supervised Graph Transformer for Deepfake Detection

> **Research Paper**: "Self-Supervised Graph Transformer for Deepfake Detection"  
> **Authors**: Aminollah Khormali and Jiann-Shiun Yuan  
> **Published**: IEEE Access 2024

A complete, production-quality, research-oriented deepfake detection platform implementing the actual Self-Supervised Graph Transformer architecture with Explainable AI.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Research Methodology](#research-methodology)
4. [Technology Stack](#technology-stack)
5. [Dataset Requirements](#dataset-requirements)
6. [Installation](#installation)
7. [Environment Configuration](#environment-configuration)
8. [Database Setup](#database-setup)
9. [Model Training](#model-training)
10. [Model Inference](#model-inference)
11. [Frontend Startup](#frontend-startup)
12. [Backend Startup](#backend-startup)
13. [API Documentation](#api-documentation)
14. [Evaluation](#evaluation)
15. [Docker Deployment](#docker-deployment)
16. [Testing](#testing)
17. [Known Limitations](#known-limitations)

---

## Project Overview

This system implements a full deepfake detection pipeline based on the Self-Supervised Graph Transformer architecture:

1. **Self-Supervised Vision Transformer** (ViT) — Feature extraction with student-teacher self-distillation and masked image modeling
2. **Graph Construction** — KNN spatial graph from 256 patch embeddings (K=8)
3. **Graph Convolutional Network** (GCN) — Learning local relationships between facial patches
4. **Graph Transformer Classifier** — 3-block, 8-head Transformer for Real/Fake classification
5. **Explainability** — Gradient-weighted attention relevancy maps

Every prediction comes from actual model inference. No hardcoded scores or simulated results.

---

## System Architecture

```
USER INPUT → Image/Video Upload → Input Validation
                                        │
                                ┌───────┴───────┐
                                │   Video?       │
                            Yes │               │ No
                                ▼               ▼
                        Frame Extraction    Face Detection
                                │               │
                                └───────┬───────┘
                                        ▼
                            Face Crop + Resize (320×320)
                                        │
                                        ▼
                            Patch Extraction (20×20 → 256)
                                        │
                                        ▼
                        Self-Supervised ViT Feature Extractor
                                        │
                                        ▼
                            Graph Construction (KNN, K=8)
                                        │
                                        ▼
                            GCN (3 layers, 256-dim)
                                        │
                                        ▼
                    Graph Transformer (3 blocks, 8 heads, 256-dim)
                                        │
                            ┌───────────┴───────────┐
                            ▼                       ▼
                    Real/Fake Classification   Relevancy Heatmap
                            │                       │
                            └───────────┬───────────┘
                                        ▼
                                Frontend Dashboard
```

---

## Research Methodology

### Self-Supervised Pretraining

The ViT backbone is pretrained using student-teacher self-distillation:

- **Student** receives masked views, trained via backpropagation
- **Teacher** receives unmasked views, updated via Exponential Moving Average (EMA)
- **L_CLS**: Cross-view CLS token self-distillation loss
- **L_MIM**: In-view masked patch self-distillation loss
- **Total**: L = L_CLS + L_MIM

### Model Architecture Parameters

| Parameter            | Value  |
|---------------------|--------|
| Input image size    | 320×320 |
| Patch size          | 20×20  |
| Total patches       | 256    |
| ViT embedding dim   | 256    |
| GCN layers          | 3      |
| KNN K value         | 8      |
| Transformer blocks  | 3      |
| Attention heads     | 8      |
| MLP hidden dim      | 512    |

---

## Technology Stack

| Layer          | Technologies |
|----------------|-------------|
| Frontend       | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, Recharts, Lucide React, Axios |
| Backend        | Python 3.11+, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0 |
| ML/AI          | PyTorch, TorchVision, timm, PyTorch Geometric, OpenCV, MTCNN, Scikit-learn |
| Database       | SQLite (dev) / PostgreSQL (prod) |
| Infrastructure | Docker, Docker Compose, FFmpeg, JWT, Pytest |

---

## Dataset Requirements

The system supports the following datasets (you must obtain access independently):

| Dataset | Access |
|---------|--------|
| FaceForensics++ | Registration required at faceforensics.com |
| Celeb-DF (V2) | Registration required |
| WildDeepfake | Registration required |
| DFDC | Registration at Kaggle |
| DeeperForensics | Registration required |
| FaceShifter | Registration required |

**Organize datasets as:**
```
datasets/
  your_dataset_name/
    train/
      real/      ← real face images
      fake/      ← fake/manipulated face images
    validation/
      real/
      fake/
    test/
      real/
      fake/
```

---

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git
- (Optional) CUDA-compatible GPU for training

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### ML Dependencies

The ML dependencies are included in `backend/requirements.txt`. For GPU support:

```bash
# CPU only (default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Key variables:

```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./deepfake.db

# JWT
SECRET_KEY=<generate-a-64-char-random-hex-string>
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Admin
FIRST_ADMIN_EMAIL=admin@deepfake.ai
FIRST_ADMIN_PASSWORD=Admin@123

# ML
DEVICE=auto
CHECKPOINTS_DIR=./checkpoints

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

---

## Database Setup

The system uses SQLite by default (auto-created on first run). For PostgreSQL:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/deepfake
```

All 10 tables are auto-created on startup via SQLAlchemy.

---

## Model Training

### Important Note

The model ships **untrained**. You must train it on your own dataset before meaningful inference is possible. The system will clearly indicate when no trained checkpoint is available.

### Step 1: Self-Supervised Pretraining

```bash
# Via the admin dashboard (POST /api/training/pretrain)
# Or programmatically:
python -c "
from ml.training.pretrain import SelfSupervisedPretrainer
trainer = SelfSupervisedPretrainer({
    'dataset_path': './datasets/your_dataset',
    'batch_size': 32,
    'epochs': 100,
    'seed': 42,
})
import asyncio
asyncio.run(trainer.train())
"
```

### Step 2: Classifier Training

```bash
# Via admin dashboard (POST /api/training/classifier)
# Or programmatically:
python -c "
from ml.training.classifier_train import GraphTransformerTrainer
trainer = GraphTransformerTrainer({
    'dataset_path': './datasets/your_dataset',
    'pretrain_checkpoint': './checkpoints/pretrain_vit_final.pt',
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'freeze_backbone': False,
    'seed': 42,
})
import asyncio
asyncio.run(trainer.train())
"
```

---

## Model Inference

Once trained, the system automatically loads the best checkpoint for inference.

### Single Image

Upload via the frontend detection page or:

```bash
curl -X POST http://localhost:8000/api/detection/image \
  -H "Authorization: Bearer <your-token>" \
  -F "file=@face_image.jpg"
```

### Video

```bash
curl -X POST http://localhost:8000/api/detection/video \
  -H "Authorization: Bearer <your-token>" \
  -F "file=@video.mp4"
```

---

## Backend Startup

```bash
cd backend
# Activate venv
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API docs are available at: `http://localhost:8000/api/docs`

---

## Frontend Startup

```bash
cd frontend
npm run dev
```

Open: `http://localhost:5173`

---

## API Documentation

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register a new user |
| `/api/auth/login` | POST | Login and get JWT |
| `/api/auth/logout` | POST | Logout |
| `/api/auth/me` | GET | Get current user profile |

### Detection
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detection/image` | POST | Upload and analyze image |
| `/api/detection/video` | POST | Upload and analyze video |
| `/api/detection/{id}` | GET | Get analysis result |
| `/api/detection/history/list` | GET | Paginated analysis history |
| `/api/detection/{id}` | DELETE | Delete analysis record |

### Model
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/info` | GET | Model architecture info |
| `/api/model/status` | GET | Model loading status |
| `/api/model/checkpoints` | GET | List checkpoints (admin) |

### Training (Admin)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/pretrain` | POST | Start ViT pretraining |
| `/api/training/classifier` | POST | Start classifier training |
| `/api/training/status/{id}` | GET | Training progress |
| `/api/training/stop/{id}` | POST | Stop training |
| `/api/training/history` | GET | Training history |

### Evaluation (Admin)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/evaluation/run` | POST | Run evaluation |
| `/api/evaluation/results` | GET | List results |
| `/api/evaluation/metrics/{id}` | GET | Detailed metrics |

### Explainability
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/explainability/{id}` | GET | Full explainability data |
| `/api/explainability/{id}/heatmap` | GET | Attention heatmap |
| `/api/explainability/{id}/graph` | GET | Graph node relevance |

---

## Evaluation

Run evaluation via the admin dashboard or API:

```bash
curl -X POST http://localhost:8000/api/evaluation/run \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_version_id": "<id>",
    "eval_type": "in_dataset",
    "dataset_path": "./datasets/test_dataset",
    "dataset_name": "FaceForensics++"
  }'
```

Supported evaluation types:
- `in_dataset` — Train and test on the same dataset
- `cross_dataset` — Train on one, test on another
- `robustness` — Test with noise, blur, compression
- `ablation` — Compare architecture variants

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up -d --build

# Access:
# Frontend: http://localhost:80
# Backend API: http://localhost:8000
# API docs: http://localhost:8000/api/docs
```

---

## Testing

```bash
# From project root
cd backend
python -m pytest ../tests/ -v --tb=short

# Or run specific test classes
python -m pytest ../tests/test_ml_pipeline.py::TestGraphBuilder -v
python -m pytest ../tests/test_ml_pipeline.py::TestFullModel -v
```

---

## Known Limitations

1. **No pretrained weights included** — You must train the model on your own dataset. The system clearly communicates when no checkpoint is available.

2. **Dataset access** — FaceForensics++, Celeb-DF, and other datasets require registration and are not automatically downloadable.

3. **Training time** — Full self-supervised pretraining (800 epochs) requires significant GPU time. Use fewer epochs for initial experiments.

4. **Video analysis** — Frame-level aggregation (mean fake probability) is an extension of the paper's frame-level architecture. The paper does not implement temporal modeling.

5. **Face detection fallback** — If MTCNN (facenet-pytorch) is not installed, falls back to OpenCV Haar cascades with lower accuracy.

6. **Evaluation metrics** — All displayed metrics come from actual model inference on test data. Never from the research paper's published results.

---

## Project Structure

```
deepfake-detection/
├── frontend/          # React + TypeScript + Vite + Tailwind
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # 11 route-level pages
│   │   ├── layouts/      # App shell layout
│   │   ├── services/     # Axios API client
│   │   ├── contexts/     # Auth context
│   │   └── types/        # TypeScript interfaces
│   └── package.json
│
├── backend/           # Python FastAPI
│   ├── app/
│   │   ├── api/          # 6 route modules
│   │   ├── core/         # Config, security, dependencies
│   │   ├── database/     # ORM models, engine
│   │   ├── schemas/      # Pydantic schemas
│   │   ├── services/     # Business logic
│   │   └── main.py
│   └── requirements.txt
│
├── ml/                # Machine Learning
│   ├── preprocessing/    # Transforms, patch extraction, video
│   ├── face_detection/   # MTCNN + Haar fallback
│   ├── feature_extractor/# ViT architecture
│   ├── self_supervised/  # Student-teacher, MIM, EMA, losses
│   ├── graph/            # KNN graph construction
│   ├── gcn/              # Graph convolutional layers
│   ├── transformer/      # Graph Transformer classifier
│   ├── explainability/   # Relevancy maps, heatmaps
│   ├── training/         # Dataset loaders, pretrain, classifier
│   ├── evaluation/       # Metrics, robustness testing
│   └── inference/        # Full pipeline, end-to-end model
│
├── tests/             # Pytest test suite
├── datasets/          # Your dataset directories
├── checkpoints/       # Model checkpoints
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## License

This project implements research methodology from the IEEE Access 2024 paper. Code is provided for educational and research purposes.

## Citation

```bibtex
@article{khormali2024selfsupervised,
  title={Self-Supervised Graph Transformer for Deepfake Detection},
  author={Khormali, Aminollah and Yuan, Jiann-Shiun},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
