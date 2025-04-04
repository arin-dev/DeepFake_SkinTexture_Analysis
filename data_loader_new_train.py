import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import dlib
import sys
import cv2
import numpy as np
# from lib.vaf_util import get_crops_landmarks
from vaf_ext import FaceTextureAnalyzer
from PIL import Image

class VideoDataset(Dataset):
    # def __init__(self, frame_direc, transform=None):
    def __init__(self, frame_direc, spatial_transform=None, temporal_transform=None):
        self.frame_direc = frame_direc
        self.subfolders = sorted(os.listdir(frame_direc))  # List of subfolders
        # self.transform = transform
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __len__(self):
        return len(self.subfolders)  # Each subfolder is one data point

    def __getitem__(self, idx):
        subfolder_path = os.path.join(self.frame_direc, self.subfolders[idx])
        frame_files = sorted(os.listdir(subfolder_path))

        # Generate consistent transform params for this batch's temporal frames
        if self.temporal_transform:
            temp_params = {
                'flip': torch.rand(1) > 0.5,
                'rotation': torch.randint(-10, 10, (1,)).item(),
                'brightness': 1 + (torch.rand(1) - 0.5) * 0.2,
                'contrast': 1 + (torch.rand(1) - 0.5) * 0.2
            }
        
        batch_images = []
        vaf_features = []

        for i in range(24*2):
            frame_path = os.path.join(subfolder_path, frame_files[i])
            if not frame_path.endswith('.jpg'):
                continue
            
            img = Image.open(frame_path)  # Changed from cv2.imread
            if img is None:
                continue
            
            # if self.transform:
            #     img = self.transform(img)

            # First 12 frames - spatial learning with random augmentations
            if i < 12*2 and self.spatial_transform:
                img = self.spatial_transform(img)
                
                # Convert transformed PIL Image back to numpy array for VAF processing
                img2 = np.array(img.permute(1, 2, 0)) * 255
                img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2BGR)
                img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                # Extract features
                analyzer = FaceTextureAnalyzer()
                features = analyzer._compute_energy(img_rgb)

                if isinstance(features, dict):
                    features = torch.from_numpy(np.array(list(features.values()), dtype=np.float32))
                    # features = torch.tensor(list(features.values()))

                vaf_features.append(features)

            # Last 12 frames - temporal learning with batch-consistent augmentations
            elif i >= 12*2 and self.temporal_transform:
                img = transforms.functional.resize(img, (128, 128))
                if temp_params['flip']:
                    img = transforms.functional.hflip(img)
                img = transforms.functional.rotate(img, temp_params['rotation'])
                img = transforms.functional.adjust_brightness(img, temp_params['brightness'])
                img = transforms.functional.adjust_contrast(img, temp_params['contrast'])
                img = transforms.functional.to_tensor(img)

            if img.shape[0] != 3 or img.shape[1] != 128 or img.shape[2] != 128:
                continue

            batch_images.append(img)

        if len(batch_images) < 24:
            return None, None, None
        
        # batch_images.append(vaf_features) ## APPENDED FEATURES AT THE END 

        video_name = '_'.join(self.subfolders[idx].split('_')[:-1])
        return torch.stack(batch_images), video_name, torch.stack(vaf_features)

def get_data_loaders(frame_direc, batch_size=1):
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    # ])
    spatial_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop(size=128, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.8, scale=(0.02, 0.20), ratio=(0.5, 2.0), inplace=True),
    ])

    # return transforms.Compose([
    #         transforms.Resize(img_size),
    #         transforms.RandomResizedCrop(size=128, scale=(0.5, 1.)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #         ], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.ToTensor(),
    #         transforms.RandomErasing(p=0.8, scale=(0.02, 0.20), ratio=(0.5, 2.0), inplace=True),
    #         transforms.Normalize(mean=mean, std=std),
    #     ])


    # dataset = VideoDataset(frame_direc, transform)
    dataset = VideoDataset(frame_direc, spatial_transform, temporal_transform=True)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
