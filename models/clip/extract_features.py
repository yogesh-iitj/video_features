"""
CLIP Video Feature Extractor with 2-Second Clips
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image  # Required import
import clip

def load_clip_model(model_name="ViT-B/32", device="auto"):
    """Load CLIP model with automatic device detection"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device

def extract_clips(video_path, clip_len=2):
    """Extract 2-second video clips with frame validation"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames_per_clip = int(fps * clip_len)
    if frames_per_clip == 0:
        return [], fps
    
    num_clips = total_frames // frames_per_clip
    clips = []
    
    for clip_idx in range(num_clips):
        clip_frames = []
        start_frame = clip_idx * frames_per_clip
        
        for offset in range(frames_per_clip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_frames.append(frame_rgb)
        
        if len(clip_frames) == frames_per_clip:
            clips.append(clip_frames)
    
    cap.release()
    return clips, fps

def process_clip(clip_frames, model, preprocess, device, batch_size=32):
    """Process clip frames and extract CLIP features"""
    try:
        features = []
        for i in range(0, len(clip_frames), batch_size):
            batch = clip_frames[i:i+batch_size]
            
            # Convert numpy arrays to PIL Images
            pil_images = [Image.fromarray(img) for img in batch]
            
            # Preprocess and batch
            preprocessed = torch.stack(
                [preprocess(img) for img in pil_images]
            ).to(device)
            
            # Extract features
            with torch.no_grad():
                batch_features = model.encode_image(preprocessed).float()
                features.append(batch_features.cpu().numpy())
        
        if features:
            return np.concatenate(features).mean(axis=0)
        return None
    
    except Exception as e:
        print(f"Clip processing error: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="CLIP Feature Extraction")
    parser.add_argument("--input_dir", required=True, 
                       help="Directory containing input videos")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for features")
    parser.add_argument("--model", default="ViT-B/32",
                       help="CLIP model variant (e.g. 'ViT-B/32', 'RN50')")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Number of frames processed simultaneously")
    parser.add_argument("--device", default="auto",
                       help="Processing device: 'cuda', 'cpu', or 'auto'")
    
    args = parser.parse_args()

    # Initialize model
    model, preprocess, device = load_clip_model(args.model, args.device)
    print(f"Initialized CLIP {args.model} on {device.upper()}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Get video files
    video_files = [f for f in os.listdir(args.input_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    print(f"Found {len(video_files)} videos for processing")

    # Process videos
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(args.input_dir, video_file)
        output_path = os.path.join(args.output_dir, f"{Path(video_file).stem}.npz")
        
        if os.path.exists(output_path):
            continue

        try:
            clips, fps = extract_clips(video_path)
            if not clips:
                continue

            # Process all clips
            clip_features = []
            for clip in clips:
                features = process_clip(clip, model, preprocess, device, args.batch_size)
                if features is not None:
                    clip_features.append(features)

            if clip_features:
                np.savez(output_path, 
                        features=np.array(clip_features),
                        fps=fps,
                        clip_length=2)
                
        except Exception as e:
            print(f"Failed to process {video_file}: {str(e)}")

    print("Feature extraction completed")

if __name__ == "__main__":
    main()
