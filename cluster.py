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
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

def extract_deep_features(img_path):
    """Extract features using CPU-only mode"""
    import tensorflow as tf
    with tf.device('/CPU:0'):  # Force CPU usage
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x, verbose=0).flatten()

def classify_frames(folder_path):
    features = []
    frame_paths = []
    
    # Get first 24 frames sorted by name
    frames = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:24]
    
    for frame_name in frames:
        frame_path = os.path.join(folder_path, frame_name)
        try:
            # Extract deep features
            feature = extract_deep_features(frame_path)
            features.append(feature)
            frame_paths.append(frame_name)
        except Exception as e:
            print(f"Skipping {frame_name}: {str(e)}")
            continue
    
    if len(features) < 2:  # Need at least 2 samples for clustering
        return None
        
    features = np.array(features)
    
    # Cluster using KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Prepare results
    return {
        'folder': os.path.basename(folder_path),
        'classification': f'cluster_{np.argmax(np.bincount(labels))}',
        'frame_details': {
            frame_paths[i]: f'cluster_{label}' 
            for i, label in enumerate(labels)
        }
    }

def process_video_frames(root_folder):
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
    with open('vaf_classification.json', 'w') as f:
        json.dump(classifications, f, indent=2)
    
    # Save to CSV
    with open('vaf_classification.csv', 'w', newline='') as f:
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