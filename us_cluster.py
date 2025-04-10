# import os
# import json
# import csv
# import numpy as np
# import argparse
# from sklearn.cluster import KMeans
# from vaf_ext import FaceTextureAnalyzer
# import cv2

# def classify_frames(folder_path):
#     features = []
#     frame_paths = []
#     analyzer = FaceTextureAnalyzer()
    
#     frames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:24]
    
#     for frame_name in frames:
#         frame_path = os.path.join(folder_path, frame_name)            
#         img = cv2.imread(frame_path)
#         if img is None:
#             continue
            
#         energy = analyzer.extract_features(img)
        
#         # Focus only on S3S3 feature (change to S3E3 if needed)
#         feat_map = energy['S3S3']
        
#         if len(feat_map.shape) == 2:
#             h, w = feat_map.shape
#             y_coords, x_coords = np.mgrid[:h, :w]
            
#             # Create spatial-feature representation
#             spatial_feat = np.column_stack([
#                 x_coords.ravel()/w,  # Normalized x
#                 y_coords.ravel()/h,   # Normalized y  
#                 feat_map.ravel()       # Feature value
#             ])
            
#             # Take mean across spatial dimensions
#             features.append(spatial_feat.mean(axis=0))
#             frame_paths.append(frame_name)
    
#     if not features:
#         return None
        
#     features = np.array(features)
    
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     labels = kmeans.fit_predict(features)
    
#     counts = np.bincount(labels)
#     majority_cluster = np.argmax(counts)
    
#     results = {
#         'folder': os.path.basename(folder_path),
#         'classification': f'cluster_{majority_cluster}',
#         'frame_details': {
#             frame_paths[i]: f'cluster_{label}' 
#             for i, label in enumerate(labels)
#         }
#     }
    
#     return results

import os
import json
import csv
import numpy as np
import argparse
from sklearn.cluster import KMeans
import cv2
import torch
import torch.nn as nn
from vaf_ext import FaceTextureAnalyzer

class FeatureProcessor(nn.Module):
    """Mimics the feature processing from models_attention.py"""
    def __init__(self):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.processor(x)

def extract_texture_features(img_paths):
    """Extract texture features using FaceTextureAnalyzer with proper CNN processing"""
    analyzer = FaceTextureAnalyzer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = FeatureProcessor().to(device)
    
    features = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        energy = analyzer.extract_features(img_rgb)
        
        feature_vector = []
        for map_name in ['L3E3', 'E3S3', 'S3S3', 'Combined']:
            feat_map = energy[map_name]
            if len(feat_map.shape) == 2:
                # Resize to match expected input size (128x128)
                feat_map = cv2.resize(feat_map, (128, 128))
                feat_tensor = torch.from_numpy(feat_map).float().unsqueeze(0).unsqueeze(0).to(device)
                
                # Process through CNN
                with torch.no_grad():
                    processed = processor(feat_tensor)
                feature_vector.extend(processed.cpu().numpy().flatten())
        
        if feature_vector:
            features.append(feature_vector)
    
    return np.array(features) if features else None

def classify_frames(folder_path):
    """Classify frames using KMeans clustering on processed features"""
    frame_paths = []
    frames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:24]
    frame_paths = [os.path.join(folder_path, f) for f in frames]
    
    features = extract_texture_features(frame_paths)
    if features is None or len(features) < 2:
        return None
        
    # Use KMeans++ initialization for better clustering
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(features)
    
    return {
        'folder': os.path.basename(folder_path),
        'classification': int(np.argmax(np.bincount(labels))),
        'frame_details': {
            frame_paths[i]: int(label) 
            for i, label in enumerate(labels)
        }
    }

def process_video_frames(root_folder):
    """Process all video frame folders in root directory"""
    classifications = {}
    csv_data = []
    
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
            
        result = classify_frames(subfolder_path)
        if result:
            classifications[subfolder] = result
            csv_data.append([subfolder, result['classification']])
    
    # Save to JSON
    with open('vaf_classification_300.json', 'w') as f:
        json.dump(classifications, f, indent=2)
    
    # Save to CSV
    with open('vaf_classification_300.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['folder', 'classification'])
        writer.writerows(csv_data)
        
    return classifications

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify video frames into clusters')
    parser.add_argument('input_folder', type=str, help='Path to folder containing video frame subfolders')
    args = parser.parse_args()
    
    print(f"Processing frames from: {args.input_folder}")
    results = process_video_frames(args.input_folder)
    print(f"Classification complete. Results saved to vaf_classification.json and vaf_classification.csv")