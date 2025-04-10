import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image
from vaf_ext import FaceTextureAnalyzer
import random

class VideoDatasetTest(Dataset):
    def __init__(self, frame_direc, transform=None, num_subfolders=None):
        self.frame_direc = frame_direc
        all_subfolders = sorted(os.listdir(frame_direc))
        
        # Select random subfolders if num_subfolders is specified
        if num_subfolders is not None and num_subfolders < len(all_subfolders):
            self.subfolders = random.sample(all_subfolders, num_subfolders)
        else:
            self.subfolders = all_subfolders
            
        self.transform = transform

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, idx):
        batch_images = []
        vaf_features = []
        subfolder_path = os.path.join(self.frame_direc, self.subfolders[idx])
        frame_files = sorted(os.listdir(subfolder_path))

        for i in range(24*2):  # Using 48 frames as in train loader
            frame_path = os.path.join(subfolder_path, frame_files[i])
            if not frame_path.endswith('.jpg'):
                continue
            
            img = Image.open(frame_path)
            if img is None:
                continue
            
            img = self.transform(img)

            if i<12*2:
                # Convert for VAF processing
                img_np = np.array(img.permute(1, 2, 0)) * 255
                img_np = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                
                # Extract VAF features
                analyzer = FaceTextureAnalyzer()
                features = analyzer.extract_features(img_rgb)
                if isinstance(features, dict):
                    features = torch.from_numpy(np.array(list(features.values()), dtype=np.float32))
                vaf_features.append(features)

            if img.shape[0] != 3 or img.shape[1] != 128 or img.shape[2] != 128:
                continue

            batch_images.append(img)

        if len(batch_images) < 24*2:
            return None, None, None

        video_name = '_'.join(self.subfolders[idx].split('_')[:-1])
        return torch.stack(batch_images), video_name, torch.stack(vaf_features)

def get_test_loaders(frame_direc, batch_size=1, num_subfolders=None):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = VideoDatasetTest(frame_direc, transform, num_subfolders)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
