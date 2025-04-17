import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def extract_texture_features(image_tensor, target_size=(256, 256)):
    """Extract features with consistent image sizing"""
    # Resize first
    resized = transforms.functional.resize(image_tensor, target_size)
    gray_image = transforms.functional.rgb_to_grayscale(resized)
        
    features = []
    for kernel_size in [3, 5]: ##, 7, 9]:
        # Gaussian blur
        blurred = transforms.functional.gaussian_blur(gray_image, kernel_size=[kernel_size, kernel_size])
        
        # Laplacian variance
        laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(image_tensor.device)
        lap_var = torch.conv2d(blurred.float(), laplacian).var()
        
        # Sobel edge detection
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to(image_tensor.device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to(image_tensor.device)
        grad_x = torch.conv2d(blurred.float(), sobel_x)
        grad_y = torch.conv2d(blurred.float(), sobel_y)
        edge_var = (grad_x.abs() + grad_y.abs()).var()
        
        features.extend([lap_var.item(), edge_var.item()])
    
    return np.array(features)

def process_images_with_clustering(folder_path, device, n_clusters=2):
    # First pass: collect all features for clustering
    all_features = []
    image_paths = []
    
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.jpg')])[:24]
            for filename in image_files:
                image_path = os.path.join(subfolder_path, filename)
                try:
                    image = read_image(image_path).to(device)
                    features = extract_texture_features(image)
                    all_features.append(features)
                    image_paths.append((subfolder, image_path))
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
    
    # Perform K-means clustering
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',  # Explicitly using k-means++
        n_init=10,         # Number of times to run with different centroid seeds
        random_state=42    # For reproducibility
    )
    clusters = kmeans.fit_predict(all_features)
    
    # Second pass: assign labels based on clusters
    results = []
    cluster_assignments = {}
    
    # Group by subfolder
    for (subfolder, _), cluster in zip(image_paths, clusters):
        if subfolder not in cluster_assignments:
            cluster_assignments[subfolder] = []
        cluster_assignments[subfolder].append(cluster)
    
    # Determine final labels for each subfolder
    for subfolder, clusters in cluster_assignments.items():
        given_label = 0 if subfolder.count("id") == 2 else 1
        cluster_counts = np.bincount(clusters, minlength=n_clusters)
        dominant_cluster = np.argmax(cluster_counts)
        
        # Assuming cluster 0 is smooth and 1 is rough (may need to reverse)
        assigned_label = dominant_cluster
        results.append((subfolder, assigned_label, given_label, cluster_counts.tolist()))
    
    return results

def main(folder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    results = process_images_with_clustering(folder_path, device)
    df = pd.DataFrame(results, columns=['Image', 'Assigned Label', 'Given Label', 'Cluster Distribution'])
    # df.to_csv('skin_texture_clustering_results_pp_ntot.csv', index=False)
    df.to_csv('skin_texture_clustering_results_pp_5.csv', index=False)

if __name__ == "__main__":
    main('/media/edward/OS/Users/arind/train_output_24_reduced')
    # main('/media/edward/OS/Users/arind/train_output_24_not_training_on_this')