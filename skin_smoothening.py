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

def classify_skin_texture(image_tensor, threshold=15.25):
    # Convert to grayscale using PyTorch
    gray_image = transforms.functional.rgb_to_grayscale(image_tensor)
    # Gaussian blur
    blurred_image = transforms.functional.gaussian_blur(gray_image, kernel_size=[3, 3])
    # Laplacian variance
    laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(image_tensor.device)
    laplacian_var = torch.conv2d(blurred_image.float(), laplacian).var()
    return laplacian_var.item(), 1 if laplacian_var > threshold else 0

def process_images_in_folder(folder_path, device):
    results = []
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        given_label = 0 if subfolder.count("id") == 2 else 1
        if os.path.isdir(subfolder_path):
            labels = []
            laplacian_values = []
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(subfolder_path, filename)
                    try:
                        image = read_image(image_path).to(device)
                        laplacian_var, label = classify_skin_texture(image)
                        labels.append(label)
                        laplacian_values.append(laplacian_var)
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
                        continue
            if labels:
                count_1 = labels.count(1)
                count_0 = labels.count(0)
                assigned_label = 1 if count_1 > count_0 else 0 if count_0 > count_1 else -1
                avg_laplacian = sum(laplacian_values)/len(laplacian_values) if laplacian_values else 0
                results.append((subfolder, assigned_label, given_label, avg_laplacian))
    return results

def main(folder_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    results = process_images_in_folder(folder_path, device)
    df = pd.DataFrame(results, columns=['Image', 'Assigned Label', 'Given Label', 'Avg Laplacian'])
    df.to_csv('skin_texture_results.csv', index=False)

if __name__ == "__main__":
    main('./cropped_output_frames')