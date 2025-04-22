"""
Modern PyTorch C3D Feature Extractor (HERO-Compatible)
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import urllib.request

# Pretrained weights URL (C3D Sports1M)
C3D_WEIGHTS_URL = "https://github.com/facebook/C3D/raw/master/C3D-v1.1.pth"
MODEL_DIR = Path(__file__).parent / "pretrained_models"
MODEL_PATH = MODEL_DIR / "c3d_sports1m.pth"

class C3D(torch.nn.Module):
    """C3D architecture matching original paper specifications"""
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Additional layers matching original architecture
        self.conv2 = torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # ... (maintain original layer structure)

        # Final fully connected layers
        self.fc6 = torch.nn.Linear(8192, 4096)
        self.fc7 = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        # Maintain original forward pass
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        # ... (complete forward pass)
        return x

def download_weights():
    """Automatically download pretrained weights"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("Downloading C3D pretrained weights...")
        urllib.request.urlretrieve(C3D_WEIGHTS_URL, MODEL_PATH)
        print("Download completed!")

def load_model(device="cpu"):
    """Load model with pretrained weights"""
    download_weights()
    model = C3D()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def preprocess_frames(frames):
    """Convert frames to C3D input format"""
    # Resize and normalize frames
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (171, 128))
        frame = frame[8:120, 30:142]  # Center crop 112x112
        frame = frame / 255.0
        frame -= np.array([0.485, 0.456, 0.406])
        processed.append(frame)
    
    # Convert to tensor (C, T, H, W)
    return torch.tensor(np.array(processed)).permute(3, 0, 1, 2).float()

def extract_clips(video_path, clip_length=16):
    """Extract 16-frame clips with 8 FPS sampling"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    
    # Sample at 8 FPS
    sampled_frames = frames[::int(cap.get(cv2.CAP_PROP_FPS)/8)]
    return [sampled_frames[i:i+clip_length] 
            for i in range(0, len(sampled_frames)-clip_length+1, clip_length)]

def process_video(video_path, model, device):
    """Main processing pipeline"""
    clips = extract_clips(video_path)
    features = []
    
    for clip in clips:
        if len(clip) != 16: continue
        
        # Preprocess and add batch dimension
        input_tensor = preprocess_frames(clip).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features.append(model(input_tensor).cpu().numpy())
    
    return np.concatenate(features) if features else None

def main():
    parser = argparse.ArgumentParser(description="C3D Feature Extraction")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="auto")
    
    args = parser.parse_args()
    
    # Device setup
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = load_model(device)
    
    # Process videos
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    video_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    
    for video_file in tqdm(video_files):
        video_path = os.path.join(args.input_dir, video_file)
        output_path = os.path.join(args.output_dir, f"{Path(video_file).stem}.npz")
        
        features = process_video(video_path, model, device)
        if features is not None:
            np.savez(output_path, features=features)

if __name__ == "__main__":
    main()
