import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import dlib
import sys
import cv2
import numpy as np
from vaf_ext import FaceTextureAnalyzer
from PIL import Image
import random

class VideoDataset(Dataset):
    def __init__(self, frame_direc, label_map, num_samples_per_class=100, spatial_transform=None, temporal_transform=None):
        self.frame_direc = frame_direc
        self.label_map = label_map
        self.num_samples_per_class = num_samples_per_class
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        
        # Group subfolders by label
        self.subfolders_by_label = {0: [], 1: []}
        for subfolder in os.listdir(frame_direc):
            video_name = '_'.join(subfolder.split('_')[:-1])
            if video_name in label_map:
                self.subfolders_by_label[label_map[video_name]].append(subfolder)
        
        # Balance the dataset
        self.balanced_subfolders = []
        for label in [0, 1]:
            random.shuffle(self.subfolders_by_label[label])
            self.balanced_subfolders.extend(self.subfolders_by_label[label][:num_samples_per_class])

    def __len__(self):
        return len(self.balanced_subfolders)

    def __getitem__(self, idx):
        subfolder_path = os.path.join(self.frame_direc, self.balanced_subfolders[idx])
        frame_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith('.jpg')])

        # Generate consistent transform params for temporal frames
        if self.temporal_transform:
            erase_h = int(128 * (0.05 + torch.rand(1) * 0.15))
            erase_w = int(erase_h * (0.5 + torch.rand(1) * 2.0))
            temp_params = {
                'flip': torch.rand(1) > 0.5,
                'rotation': torch.randint(-10, 10, (1,)).item(),
                'brightness': 1 + (torch.rand(1) - 0.5) * 0.2,
                'contrast': 1 + (torch.rand(1) - 0.5) * 0.2,
                'erase': torch.rand(1) < 0.5,
                'erase_i': torch.randint(0, 128 - erase_h, (1,)).item(),
                'erase_j': torch.randint(0, 128 - erase_w, (1,)).item(),
                'erase_h': erase_h,
                'erase_w': erase_w
            }
        
        batch_images = []
        vaf_features = []

        for i, frame_file in enumerate(frame_files[:24*2]):
            frame_path = os.path.join(subfolder_path, frame_file)
            img = Image.open(frame_path)
            if img is None:
                continue
            
            # First 12 frames - spatial learning with random augmentations
            if i < 12*2 and self.spatial_transform:
                img = self.spatial_transform(img)
                img2 = np.array(img.permute(1, 2, 0)) * 255
                img_rgb = cv2.cvtColor(cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
                
                analyzer = FaceTextureAnalyzer()
                features = analyzer.extract_features(img_rgb)
                if isinstance(features, dict):
                    features = torch.from_numpy(np.array(list(features.values()), dtype=np.float32))
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

                if temp_params['erase']:
                    img = transforms.functional.erase(
                        img,
                        i=temp_params['erase_i'],
                        j=temp_params['erase_j'],
                        h=temp_params['erase_h'],
                        w=temp_params['erase_w'],
                        v=0,
                        inplace=False
                    )
                batch_images.append(img)

        if len(batch_images) < 24:
            return None, None, None, None
        
        video_name = '_'.join(self.balanced_subfolders[idx].split('_')[:-1])
        label = self.label_map[video_name]
        return torch.stack(batch_images), video_name, torch.stack(vaf_features), label

def get_data_loaders(frame_direc, label_map, num_samples_per_class=100, batch_size=1):
    spatial_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop(size=128, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.10), ratio=(0.5, 2.0), inplace=True),
    ])

    dataset = VideoDataset(
        frame_direc=frame_direc,
        label_map=label_map,
        num_samples_per_class=num_samples_per_class,
        spatial_transform=spatial_transform,
        temporal_transform=True
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)