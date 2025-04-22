# Video Feature Extraction

This repository contains a collection of scripts for extracting features from videos using different deep learning models such as SlowFast and CLIP.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── raw_videos/            # Directory containing input videos
├── features/             # Directory for extracted features
└── models/
    ├── slow_fast/        # SlowFast feature extraction scripts
    └── clip/             # CLIP feature extraction scripts
```

## Installation

Set up the environment using conda:

```bash
# Create and activate conda environment
conda create -n video_features python=3.8
conda activate video_features

# Install PyTorch dependencies
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install additional dependencies
pip install pytorchvideo opencv-python tqdm
pip install git+https://github.com/openai/CLIP.git
pip install decord gluoncv-torch torch tqdm
```

## Usage

### SlowFast Feature Extraction

The SlowFast model is used for extracting spatiotemporal features from videos. It uses two pathways:
- A slow pathway that captures spatial semantics
- A fast pathway that captures motion at fine temporal resolution

To extract features using the SlowFast model:

```bash
python models/slow_fast/extract_features.py --input_dir ./raw_videos/ --output_dir ./features/ --clip_len 2
```

Parameters:
- `--input_dir`: Directory containing input videos
- `--output_dir`: Directory where extracted features will be saved
- `--clip_len`: Length of each clip segment in seconds (default: 2)

The script processes each video by:
1. Splitting it into 2-second clips
2. Processing each clip through the SlowFast model
3. Extracting features
4. Saving the features in .npz format (one file per video)

### CLIP Feature Extraction

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. This repository uses CLIP's vision encoder to extract visual features from video frames.

To extract features using the CLIP model:

```bash
python models/clip/extract_features.py --input_dir ./raw_videos/ --output_dir ./features/ --model ViT-B/32 --batch_size 32
```

Parameters:
- `--input_dir`: Directory containing input videos
- `--output_dir`: Directory where extracted features will be saved
- `--model`: CLIP model variant (default: 'ViT-B/32', alternatives: 'RN50', etc.)
- `--batch_size`: Number of frames processed simultaneously (default: 32)
- `--device`: Processing device: 'cuda', 'cpu', or 'auto' (default: 'auto')

The script processes each video by:
1. Extracting 2-second clips from the video
2. Processing each frame through CLIP's image encoder
3. Averaging features across frames in each clip
4. Saving the features in .npz format (one file per video)

## Technical Details

### SlowFast Feature Extractor

The SlowFast feature extractor uses the pre-trained SlowFast R50 model from PyTorchVideo with the following specifications:

- **Model Architecture**: SlowFast R50 (ResNet-50 backbone)
- **Feature Dimension**: The extracted features have a dimension of `[n, 2304]`
  - `n`: number of clips in the video
  - `2304`: feature dimensionality
- **Processing Pipeline**:
  - Video is split into 2-second clips
  - Each clip is processed through a dual pathway network:
    - Slow pathway: Operates at 1/4 temporal resolution to capture spatial semantics
    - Fast pathway: Operates at full temporal resolution to capture motion
  - The classification layer is removed to obtain feature embeddings
  - Frames are standardized with mean=[0.45, 0.45, 0.45] and std=[0.225, 0.225, 0.225]
  - Short side of frames is scaled to 256 pixels

The SlowFast architecture is based on the paper [SlowFast Networks for Video Recognition (ICCV 2019)](https://arxiv.org/abs/1812.03982) by Christoph Feichtenhofer et al.

### CLIP Feature Extractor

The CLIP feature extractor uses OpenAI's CLIP model with the following specifications:

- **Model Architecture**: By default, ViT-B/32 (Vision Transformer)
- **Feature Dimension**: The extracted features have a dimension of `[num_clips, 512]`
  - `num_clips`: number of 2-second clips in the video
  - `512`: feature dimensionality (may vary based on model variant)
- **Processing Pipeline**:
  - Video is split into 2-second clips
  - Each frame is processed using CLIP's official augmentations
  - Features are extracted from CLIP's image encoder
  - Features are averaged across all frames in a clip
  - Additional metadata (fps, clip length) is saved alongside features

The output .npz files contain:
- `features`: Feature vectors for each clip
- `fps`: Original video frame rate
- `clip_length`: Length of each clip (2 seconds)

CLIP is based on the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford et al.

## Output Format

The extracted features are saved as NumPy `.npz` files in the output directory:

### SlowFast Output
```
{video_name}.npz:
  - features: array of shape [num_clips, 2304]
```

### CLIP Output
```
{video_name}.npz:
  - features: array of shape [num_clips, 512]
  - fps: original video frame rate
  - clip_length: 2 (seconds)
```

## References

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [PyTorchVideo](https://pytorchvideo.org/)
- [OpenAI CLIP](https://github.com/openai/CLIP)