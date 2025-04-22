# Video Feature Extraction

This repository contains a collection of scripts for extracting features from videos using different deep learning models such as SlowFast and CLIP.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── raw_video/            # Directory containing input videos
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
python models/slow_fast/extract_features.py --input_dir ./raw_video/ --output_dir ./features/ --clip_len 2
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



## References

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [PyTorchVideo](https://pytorchvideo.org/)
- [CLIP: Connecting Text and Images](https://openai.com/research/clip)


