dependecies:
conda create -n slowfast python=3.8
conda activate slowfast
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install pytorchvideo opencv-python tqdm
pip install git+https://github.com/openai/CLIP.git
pip install decord gluoncv-torch torch tqdm

slow_fast feature scripts:
extract_features.py
import os
import sys
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda

# Torchvision compatibility fix

import torchvision.transforms.functional as F_t
sys.modules['torchvision.transforms.functional_tensor'] = F_t

from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.transforms import (
UniformTemporalSubsample,
ShortSideScale,
Normalize
)

class PackPathway(torch.nn.Module):
def __init__(self, alpha=4):
super().__init__()
self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        # Add batch dimension [1, C, T, H, W]
        frames = frames.unsqueeze(0)
        
        # Slow pathway (1/4 temporal resolution)
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(0, frames.shape[2]-1, frames.shape[2]//self.alpha).long()
        )
        return [slow_pathway, frames]  # [slow, fast]
    def load_model():
"""Load modified SlowFast model"""
model = slowfast_r50(pretrained=True)
model.blocks[6].proj = torch.nn.Identity()  \# Remove classification layer
return model.eval().cpu()

def preprocess_video(video_path, clip_len=2):
"""Process video into 2-second clips with proper dimensions"""
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate clip parameters
    frames_per_clip = int(fps * clip_len)
    num_clips = total_frames // frames_per_clip
    
    clips = []
    for clip_idx in range(num_clips):
        clip_frames = []
        start_frame = clip_idx * frames_per_clip
        
        # Extract frames for this clip
        for frame_offset in range(frames_per_clip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_offset)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (256, 256))
            clip_frames.append(frame)
        
        if len(clip_frames) == frames_per_clip:
            # Convert to tensor [C, T, H, W]
            clip_tensor = torch.tensor(np.array(clip_frames)).permute(3, 0, 1, 2).float()
            clips.append(clip_tensor)
    
    cap.release()
    return clips
    def process_clip(clip, clip_len):
"""Process individual clip through transformation pipeline"""
transform = Compose([
UniformTemporalSubsample(32 if clip_len == 2 else 64),  \# 16fps for 2sec
Lambda(lambda x: x/255.0),
Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
ShortSideScale(size=256),
PackPathway(alpha=4)
])
return transform(clip)

def extract_features(model, processed_clips):
"""Extract features from all clips"""
features = []
with torch.no_grad():
for clip in processed_clips:
\# Input: list of [slow_pathway, fast_pathway]
\# Each pathway shape: [1, C, T, H, W]
output = model(clip)
features.append(output.squeeze().cpu().numpy())
return np.array(features)

def main():
parser = argparse.ArgumentParser(description="SlowFast Feature Extractor")
parser.add_argument("--input_dir", required=True, help="Input video directory")
parser.add_argument("--output_dir", required=True, help="Output feature directory")
parser.add_argument("--clip_len", type=int, default=2, help="Clip length in seconds")
args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model()
    
    video_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(args.input_dir, video_file)
        output_path = os.path.join(args.output_dir, f"{Path(video_file).stem}.npz")
        
        if os.path.exists(output_path):
            continue
            
        try:
            # 1. Split video into 2-second clips
            clips = preprocess_video(video_path, args.clip_len)
            
            # 2. Process each clip through SlowFast
            processed_clips = [process_clip(c, args.clip_len) for c in clips]
            
            # 3. Extract features
            features = extract_features(model, processed_clips)
            
            # 4. Save features
            np.savez(output_path, features=features)
            
        except Exception as e:
            print(f"Failed {video_file}: {str(e)}")
    if __name__ == "__main__":
main()

similarly we have scripts for clip in a different folder

Here's a minimal README.md tailored to your setup:

```markdown
# Video Feature Extraction

Extract video features using SlowFast and CLIP models.

## Installation

```

conda create -n slowfast python=3.8
conda activate slowfast
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install pytorchvideo opencv-python tqdm
pip install git+https://github.com/openai/CLIP.git

```

## Usage

### Folder Structure
```

video_features/
├── raw_video/       \# Put input videos here
├── features/        \# Extracted features will be saved here
├── slow_fast/       \# Contains SlowFast extraction script
└── clip/            \# Contains CLIP extraction script

```

### SlowFast Feature Extraction
Extract features using SlowFast model (2-second clips by default):

```

python slow_fast/extract_features.py \
--input_dir ./raw_video \
--output_dir ./features/slowfast \
--clip_len 2

```

**Arguments**:
- `--input_dir`: Directory containing input videos
- `--output_dir`: Directory to save extracted features (.npz files)
- `--clip_len`: Clip length in seconds (default: 2)

### CLIP Feature Extraction
Extract features using CLIP model:

```

python clip/extract_features.py \
--input_dir ./raw_video \
--output_dir ./features/clip \
--clip_len 2 \
--batch_size 32

```

**Arguments**:
- `--batch_size`: Number of frames processed at once (default: 32)
- `--model`: CLIP model variant (default: ViT-B/32)

## Output Format
Features are saved as `.npz` files containing:
- `features` array: Shape (num_clips, feature_dim)
- Metadata including timestamps and FPS

**Note**: Scripts automatically skip already processed videos.
```


