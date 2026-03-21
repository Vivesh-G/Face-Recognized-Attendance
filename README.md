# Part-fViT Attendance System

A real-time face recognition-based attendance system powered by **Part-fViT (Landmark-aware Facial Vision Transformer)** with **EdgeFace** for landmark detection. This system uses facial landmarks for accurate face recognition and automatic attendance marking via a web interface.

## Features

- **Real-time Face Detection** - Uses MTCNN for accurate face detection
- **EdgeFace Landmark Detection** - EdgeFace XS model for efficient facial landmark prediction
- **Part-fViT Recognition** - Landmark-aware Vision Transformer for robust face recognition
- **Live Tracking** - Real-time webcam-based attendance marking
- **Web Interface** - Modern HTML/CSS interface for easy operation
- **Student Enrollment** - Easy registration of students with face images
- **CSV Export** - Export attendance records to CSV
- **Cosine Similarity Matching** - Efficient embedding comparison

## Architecture

### Part-fViT Model

The system uses **Part-fViT (Landmark-aware Facial Vision Transformer)** which consists of:

1. **EdgeFace** (`LAFS/PartfVit.py`)
   - Loaded via torch.hub from `otroshi/edgeface`
   - Model: `edgeface_xs_gamma_06`
   - Outputs 512-dimensional embeddings
   - Used for landmark feature extraction
   - Efficient architecture for real-time processing

2. **Landmark Coordinate Prediction** (`LAFS/PartfVit.py`)
   - Output layer maps 512-D EdgeFace embeddings to 196 landmark coordinates (14×14 grid)
   - Uses learnable linear layer with dropout

3. **Landmark Patch Extraction** (`LAFS/utils.py`)
   - Extracts 196 patches at predicted landmark locations
   - Uses grid sampling for differentiable operation

4. **Vision Transformer** (`LAFS/PartfVit.py`)
   - 12-layer transformer with 11 attention heads
   - 768-dimensional embedding output
   - Processes landmark-based patches for face recognition

```
Input Face Image (112×112)
         ↓
   EdgeFace XS (512-D embeddings)
         ↓
   Landmark Coordinate Prediction (196 landmarks)
         ↓
   Landmark-based Patch Extraction (196 patches)
         ↓
   Vision Transformer (12 layers, 11 heads)
         ↓
   Face Embedding (768-dim)
```

## Installation

### Install Dependencies

```bash
uv iniit
uv sync
```

### Model Checkpoint

The system uses pretrained EdgeFace by default. Optionally, you can add a custom checkpoint:
- Place `lafs_webface_finetune_withaugmentation.pth` in the project root
- Download from: https://drive.google.com/file/d/1BUYm2Bcgp8ZRlBcwOZxiJtWiQAvK2Ujy/view

## Quick Start

### Run the Server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Web Interface

Open your browser at: `http://localhost:8000`

## Directory Structure

```
├── main.py                 # FastAPI application & routes
├── attendance_manager.py  # CSV attendance management
├── pyproject.toml         # Project dependencies
├── requirements.txt       # pip requirements
├── dataset/               # Face images for enrollment
│   └── <person_name>/
│       └── <images>.jpg
├── embeddings/            # Stored face embeddings
├── static/
│   ├── index.html         # Web interface
│   └── style.css          # Styles
├── LAFS/
│   ├── face_engine.py     # Face recognition engine
│   ├── PartfVit.py       # Part-fViT model with EdgeFace
│   ├── Mobilenet.py      # MobileNet backbone (legacy)
│   └── utils.py           # Utility functions
```

## Dataset Organization

For bulk enrollment, organize images in the `dataset/` folder:

```
dataset/
├── John_Doe/
│   ├── John_Doe_01__ClassA__DeptA.jpg
│   └── John_Doe_02__ClassA__DeptA.jpg
└── Jane_Smith/
    └── Jane_Smith_01__ClassB__DeptB.jpg
```

Filename format: `<Name>_<Number>__<Class>__<Department>.<ext>`

Then click **"Rebuild DB"** in the web interface to process all images.

## Technical Details

### Preprocessing
- Resize to 112×112
- Normalize with mean=0.5, std=0.5

### EdgeFace Integration
- Loaded dynamically via `torch.hub.load('otroshi/edgeface', 'edgeface_xs_gamma_06')`
- Provides 512-dimensional embeddings for landmark feature extraction
- Fallback to pretrained weights if custom checkpoint unavailable

### Embedding Extraction
- Forward pass through Part-fViT
- L2 normalization for cosine similarity

## License

This project is for educational and research purposes.

## Acknowledgments

```
@article{edgeface,
  title={Edgeface: Efficient face recognition model for edge devices},
  author={George, Anjith and Ecabert, Christophe and Shahreza, Hatef Otroshi and Kotwal, Ketan and Marcel, Sebastien},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2024}
}
```
```
@InProceedings{Sun_2024_CVPR,
    author    = {Sun, Zhonglin and Feng, Chen and Patras, Ioannis and Tzimiropoulos, Georgios},
    title     = {LAFS: Landmark-based Facial Self-supervised Learning for Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```
