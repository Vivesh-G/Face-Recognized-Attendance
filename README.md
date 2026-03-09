# Part-fViT Attendance System

A real-time face recognition-based attendance system powered by **Part-fViT (Landmark-aware Facial Vision Transformer)**. This system uses facial landmarks for accurate face recognition and automatic attendance marking.

## Features

- **Real-time Face Detection** - Uses MTCNN for accurate face detection
- **Part-fViT Recognition** - Landmark-aware Vision Transformer for robust face recognition
- **Live Tracking** - Real-time webcam-based attendance marking
- **Student Registration** - Easy registration of students with face images
- **Cosine Similarity Matching** - Efficient embedding comparison

## Architecture

### Part-fViT Model

The system uses **Part-fViT (Landmark-aware Facial Vision Transformer)** which consists of:

1. **MobileNetV3 Backbone** (`LAFS/Mobilenet.py`)
   - Landmark detection network
   - Outputs 160-dimensional features
   - Efficient architecture for real-time processing

2. **Landmark Patch Extraction** (`LAFS/utils.py`)
   - Extracts 196 patches (14×14 grid) at predicted landmark locations
   - Uses grid sampling for differentiable operation

3. **Vision Transformer** (`LAFS/PartfVit.py`)
   - 12-layer transformer with 11 attention heads
   - 768-dimensional embedding output
   - Processes landmark-based patches for face recognition

```
Input Face Image (112×112)
         ↓
   MobileNetV3 (STN)
   Landmark Prediction
         ↓
   Landmark-based Patch Extraction (196 patches)
         ↓
   Vision Transformer (12 layers, 11 heads)
         ↓
   Face Embedding (768-dim)
```

## Installation



### Model Checkpoint

The system requires a pre-trained Part-fViT checkpoint:
- Place `lafs_webface_finetune_withaugmentation.pth` in the project root from the original Author [arxiv](https://arxiv.org/abs/2403.08161)
- [Download Here](https://drive.google.com/file/d/1BUYm2Bcgp8ZRlBcwOZxiJtWiQAvK2Ujy/view)

## Technical Details

### Preprocessing
- Convert BGR to RGB
- Resize to 112×112
- Normalize with mean=0.5, std=0.5

### Embedding Extraction
- Forward pass through Part-fViT
- L2 normalization for cosine similarity

### Checkpoint Loading
Supports multiple checkpoint formats:
- `teacher` key ( distillation models)
- `model` key
- Direct state dict

Handles key prefixes: `module.`, `backbone.`, `encoder.`

## Performance

- **Face Detection**: MTCNN (real-time)
- **Recognition**: Part-fViT (~30 FPS on GPU)
- **Embedding Dimension**: 768

## License

This project is for educational and research purposes.

## Acknowledgments

- Part-fViT: Landmark-aware Facial Vision Transformer ([Link](https://github.com/szlbiubiubiu/LAFS_CVPR2024)), [arxiv](https://arxiv.org/abs/2403.08161)