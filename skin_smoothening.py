# import cv2
# import numpy as np
# import os
# import pandas as pd
# import torch

# def classify_skin_texture(image, threshold=15.25):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
#     laplacian_var = cv2.Laplacian(blurred_image, cv2.CV_64F).var()
#     return 1 if laplacian_var > threshold else 0

# def process_images_in_folder(folder_path):
#     results = []
#     for subfolder in os.listdir(folder_path):
#         subfolder_path = os.path.join(folder_path, subfolder)
#         given_label = 0 if subfolder.count("id") == 2 else 1
#         if os.path.isdir(subfolder_path):
#             labels = []
#             for filename in os.listdir(subfolder_path):
#                 if filename.endswith('.jpg'):
#                     image_path = os.path.join(subfolder_path, filename)
#                     image = cv2.imread(image_path)
#                     if image is None:
#                         print(f"Error: Could not open image {image_path}.")
#                         continue
#                     label = classify_skin_texture(image)
#                     labels.append(label)
#             if labels:
#                 count_1 = labels.count(1)
#                 count_0 = labels.count(0)
#                 if count_1 > count_0:
#                     assigned_label = 1
#                 elif count_0 > count_1:
#                     assigned_label = 0
#                 else:
#                     assigned_label = -1
#                 # results.append((subfolder, assigned_label, count_1, count_0, given_label))
#                 results.append((subfolder, assigned_label, given_label))
#     return results

# def main(folder_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Running on device: {device}")
#     if device.type == 'cuda':
#         print(f"GPU Name: {torch.cuda.get_device_name(0)}")
#         print(f"Memory Allocated: {torch.cuda.memory_allocated(0)} bytes")
#         print(f"Memory Cached: {torch.cuda.memory_reserved(0)} bytes")
    
#     results = process_images_in_folder(folder_path)
#     df = pd.DataFrame(results, columns=['Image', 'Assigned Label', 'Count 1', 'Count 0', 'Given Label'])
#     df.to_csv('skin_texture_results.csv', index=False)

# if __name__ == "__main__":
#     main('./cropped_output_frames')

import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
import pandas as pd
import numpy as np

def classify_skin_texture(image_tensor, threshold=9.5):
    """Improved skin texture classification using multiple features"""
    # Convert to grayscale
    gray_image = transforms.functional.rgb_to_grayscale(image_tensor)
    
    # Multi-scale analysis
    features = []
    for kernel_size in [3, 5, 7]:
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
    
    # Combine features with weighted average
    combined_score = 0.6*features[0] + 0.3*features[2] + 0.1*features[4]  # weights for laplacian at different scales
    return combined_score, 1 if combined_score > threshold else 0

def process_images_in_folder(folder_path, device):
    results = []
    for subfolder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, subfolder)
        given_label = 0 if subfolder.count("id") == 2 else 1
        if os.path.isdir(subfolder_path):
            labels = []
            texture_scores = []
            # Get first 24 images sorted by name
            image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.jpg')])[:24]
            
            for filename in image_files:
                image_path = os.path.join(subfolder_path, filename)
                try:
                    image = read_image(image_path).to(device)
                    score, label = classify_skin_texture(image)
                    labels.append(label)
                    texture_scores.append(score)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
            
            if labels:
                count_1 = labels.count(1)
                count_0 = labels.count(0)
                assigned_label = 1 if count_1 > count_0 else 0 if count_0 > count_1 else -1
                avg_score = np.mean(texture_scores) if texture_scores else 0
                results.append((subfolder, assigned_label, given_label, avg_score))
    return results

def main(folder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    results = process_images_in_folder(folder_path, device)
    df = pd.DataFrame(results, columns=['Image', 'Assigned Label', 'Given Label', 'Texture Score'])
    df.to_csv('skin_texture_results_6.csv', index=False)

if __name__ == "__main__":
    main('/media/edward/OS/Users/arind/train_output_24_reduced')
    # main('/media/edward/OS/Users/arind/train_output_24_not_training_on_this')

# import torch
# import torchvision.transforms as transforms
# from torchvision.io import read_image
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt  # For optional visualization

# def classify_skin_texture(image_tensor):
#     """Improved skin texture classification using multiple features (returns raw score)"""
#     # Convert to grayscale
#     gray_image = transforms.functional.rgb_to_grayscale(image_tensor)
    
#     # Multi-scale analysis
#     features = []
#     for kernel_size in [3, 5, 7]:
#         # Gaussian blur
#         blurred = transforms.functional.gaussian_blur(gray_image, kernel_size=[kernel_size, kernel_size])
        
#         # Laplacian variance
#         laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(image_tensor.device)
#         lap_var = torch.conv2d(blurred.float(), laplacian).var()
        
#         # Sobel edge detection
#         sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to(image_tensor.device)
#         sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to(image_tensor.device)
#         grad_x = torch.conv2d(blurred.float(), sobel_x)
#         grad_y = torch.conv2d(blurred.float(), sobel_y)
#         edge_var = (grad_x.abs() + grad_y.abs()).var()
        
#         features.extend([lap_var.item(), edge_var.item()])
    
#     # Combine features with weighted average
#     return 0.6*features[0] + 0.3*features[2] + 0.1*features[4]  # weights for laplacian at different scales

# def analyze_score_distribution(folder_path, device, sample_size=5276, visualize=True, sensitivity=0.05):
#     """
#     Analyze texture scores with adjustable sensitivity (0-1)
#     0 = most conservative (higher threshold), 1 = most aggressive (lower threshold)
#     """
#     scores = []
    
#     # Collect samples
#     for subfolder in sorted(os.listdir(folder_path))[:sample_size]:
#         subfolder_path = os.path.join(folder_path, subfolder)
#         if os.path.isdir(subfolder_path):
#             image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.jpg')])[:10]
#             for filename in image_files:
#                 try:
#                     image = read_image(os.path.join(subfolder_path, filename)).to(device)
#                     score = classify_skin_texture(image)
#                     scores.append(score)
#                 except Exception as e:
#                     continue
    
#     if not scores:
#         print("Warning: No scores collected, using default threshold 12.0")
#         return 12.0  # Balanced default
    
#     scores = np.array(scores)
    
#     # Calculate multiple threshold candidates
#     candidates = {
#         'percentile_80': np.percentile(scores, 80),
#         'percentile_85': np.percentile(scores, 85),
#         'mean_1.5std': np.mean(scores) + 1.5 * np.std(scores),
#         'mean_2.0std': np.mean(scores) + 2.0 * np.std(scores),
#         'bimodal': find_bimodal_threshold(scores) if len(scores) > 100 else None
#     }
    
#     # Filter out None values and sort candidates
#     valid_candidates = sorted([v for v in candidates.values() if v is not None])
    
#     if not valid_candidates:
#         return 12.0
    
#     # Select threshold based on sensitivity (0-1)
#     idx = min(int(len(valid_candidates) * sensitivity), len(valid_candidates)-1)
#     threshold = valid_candidates[idx]
    
#     if visualize:
#         plt.figure(figsize=(10, 5))
#         plt.hist(scores, bins=50, alpha=0.7)
#         plt.title(f"Texture Score Distribution (Threshold: {threshold:.2f})")
#         plt.xlabel("Score")
#         plt.ylabel("Frequency")
        
#         for name, value in candidates.items():
#             if value is not None:
#                 plt.axvline(x=value, linestyle='--', alpha=0.5, 
#                            label=f"{name}: {value:.2f}")
#         plt.axvline(x=threshold, color='red', label=f"Selected: {threshold:.2f}")
#         plt.legend()
#         plt.show()
    
#     print("Threshold candidates:")
#     for name, value in sorted(candidates.items(), key=lambda x: x[1] or 0):
#         if isinstance(value, (int, float)):
#             print(f"{name:>12}: {value:.2f}")
#         else:
#             print(f"{name:>12}: {value or 'N/A'}")
    
#     print(f"\nSelected threshold: {threshold:.2f} (sensitivity: {sensitivity})")
#     return float(threshold)

# def find_bimodal_threshold(scores):
#     """Helper function for bimodal threshold detection"""
#     try:
#         hist, bin_edges = np.histogram(scores, bins=50)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
#         # Find major peaks
#         peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
#         if len(peaks) >= 2:
#             sorted_peaks = sorted(peaks, key=lambda x: hist[x], reverse=True)[:2]
#             valley = np.argmin(hist[sorted_peaks[0]:sorted_peaks[1]]) + sorted_peaks[0]
#             return bin_centers[valley]
#     except:
#         return None

# def process_images_in_folder(folder_path, device):
#     """Process all images in folder using automatically determined threshold"""
#     # Determine optimal threshold
#     threshold = analyze_score_distribution(folder_path, device, visualize=True)
#     print(f"Using automatically determined threshold: {threshold:.2f}")
    
#     results = []
#     for subfolder in sorted(os.listdir(folder_path)):
#         subfolder_path = os.path.join(folder_path, subfolder)
#         given_label = 0 if subfolder.count("id") == 2 else 1
#         if os.path.isdir(subfolder_path):
#             labels = []
#             texture_scores = []
#             # Get first 24 images sorted by name
#             image_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.jpg')])[:24]
            
#             for filename in image_files:
#                 image_path = os.path.join(subfolder_path, filename)
#                 try:
#                     image = read_image(image_path).to(device)
#                     score = classify_skin_texture(image)
#                     label = 1 if score > threshold else 0
#                     labels.append(label)
#                     texture_scores.append(score)
#                 except Exception as e:
#                     print(f"Error processing {image_path}: {str(e)}")
#                     continue
            
#             if labels:
#                 count_1 = labels.count(1)
#                 count_0 = labels.count(0)
#                 assigned_label = 1 if count_1 > count_0 else 0 if count_0 > count_1 else -1
#                 avg_score = np.mean(texture_scores) if texture_scores else 0
#                 results.append((subfolder, assigned_label, given_label, avg_score, threshold))
#     return results

# def main(folder_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Running on device: {device}")
#     if device.type == 'cuda':
#         print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
#     results = process_images_in_folder(folder_path, device)
#     df = pd.DataFrame(results, columns=['Image', 'Assigned Label', 'Given Label', 'Texture Score', 'Threshold'])
#     output_file = 'skin_texture_results_auto_threshold.csv'
#     df.to_csv(output_file, index=False)
#     print(f"Results saved to {output_file}")

# if __name__ == "__main__":
#     main('/media/edward/OS/Users/arind/train_output_24_reduced')