import torch
import torch.optim as optim
import torch.nn as nn
from models_attention import TwoStreamNetworkTransferLearning
from data_loader_new_train import get_data_loaders
# from vaf_ext.py import 
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_device_for_model():
    # Check memory usage on all available GPUs
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(i)
        mem_total = torch.cuda.get_device_properties(i).total_memory
        if mem_alloc < mem_total * 0.95:
            return torch.device(f'cuda:{i}')
    raise MemoryError("All GPUs are out of memory!")

def save_input_images(data, video_name):
    """Save input images for verification"""
    print("Saving input images for verification")
    os.makedirs(f'input_images_{video_name}', exist_ok=True)
    data = data.cpu().numpy()  # Convert to numpy [1,24,3,128,128]
    
    for frame_idx in range(data.shape[1]):  # 24 frames
        img = data[0, frame_idx].transpose(1, 2, 0)  # [H,W,C] from [C,H,W]
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(f'input_images_{video_name}/frame_{frame_idx}.png', img)

# def visualize_vaf_features(vaf_features, video_name):
#     print("Inside Visualizer")

#     # Save raw features for local visualization
#     features = vaf_features.cpu().numpy()  # Take first in batch [24,4,128,128,3]
    
#     # Save each feature map as separate images
#     os.makedirs(f'vaf_vis_{video_name}', exist_ok=True)
#     for frame_idx in range(features.shape[0]):  # 24 frames
#         for feat_idx in range(features.shape[1]):  # 4 features
#             img = features[frame_idx, feat_idx]  # [128,128,3]
#             img = (img * 255).astype(np.uint8)  # Convert to 0-255 range
#             cv2.imwrite(f'vaf_vis_{video_name}/frame_{frame_idx}_feat_{feat_idx}.png', img)

def train_model(num_epochs, frame_direc, device, batch_size=1, model_name=None, start_epoch=0):
    print("Entering to train data!")
    model = TwoStreamNetworkTransferLearning()

    # Use DataParallel with all available GPUs
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    # device = get_device_for_model()
    model = model.to(device)
    
    # Load pre-trained model if path is provided
    checkpoint_path = f'models/{model_name}_model_epoch_{start_epoch}.pth'
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001) ## reduced lr from 0.001
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle DataParallel state dict
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    
    criterion = nn.BCELoss()

    print("Opening Label.json!")
    # with open('/kaggle/input/train-data-new/labels.json', 'r') as file:
    with open('new_training_set.json', 'r') as file:
    # with open('OLD_12/real_labels_12.json', 'r') as file:
        label_map = json.load(file)

    model.train()

    # Add gradient accumulation for larger effective batch size
    accumulation_steps = 4
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        train_loader = get_data_loaders(frame_direc, batch_size=batch_size)
        
        batch_count = 0
        optimizer.zero_grad()
        
        for i, (data, video_names, vaf_features) in enumerate(train_loader):
            if data is None:
                print("No valid data returned from loader, skipping this batch.")
                continue

            # print(video_names)
            # print(type(data), data.shape)
            # print(type(vaf_features), len(vaf_features), len(vaf_features[0]))

            # data = data[:, 12*2:, :, :, :]  # Reduces to [1,24,3,128,128]
            # save_input_images(data, video_names[0])  # Save input images for verification
            
            # print(vaf_features.shape)
            # vaf_features = vaf_features.view(-1, 24, 4)
            # print("data shape:", data.shape)
            # print("vaf_features shape:", vaf_features.shape)
            data = data.float().to(device)
            vaf_features = vaf_features.float().to(device)

            # Visualize VAF features
            # visualize_vaf_features(vaf_features[0], video_names[0])

            # # Ensure VAF features are [batch_size, 24, 4]
            # if vaf_features.dim() == 2:  # If flattened
            #     vaf_features = vaf_features.view(-1, 24, 4)

            # print("data shape:", data.shape)
            # print("vaf_features shape:", vaf_features.shape)

            outputs = model(data, vaf_features).squeeze(1)
            # outputs = model(data, vaf_features).squeeze(1)
            
            # print(video_names)

            labels_tensor = torch.tensor([
                0 if label_map.get(video_name) == -1 else label_map.get(video_name)
                for video_name in video_names
            ]).float().to(device)

            loss = criterion(outputs, labels_tensor)
            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if (batch_count + 1) % 40 == 0 or (batch_count+1) == len(train_loader):
                epoch_time = time.time() - start_time
                start_time = time.time()
                print(f'Batch {batch_count+1}/{len(train_loader)} // {epoch+1}/{num_epochs}, Loss: {loss.item()}, Time: {epoch_time}')
            batch_count += 1
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        
        # Save checkpoint every epoch with proper DataParallel handling
        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            
            # save_path = f'/models/{model_name}_model_epoch_{epoch+1}.pth'
            save_path = f'models/{model_name}_model_epoch_{epoch+1}.pth'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(checkpoint, save_path)
            print(f'Checkpoint saved at epoch {epoch+1} to {save_path}')


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        print("GPU cache cleared")
    else:
        print("No GPU available")

# Usage

def main():
    num_epochs = 35
    # frame_direc = '/kaggle/input/train-output-24-reduced/train_output_24_reduced'
    # frame_direc = '/media/edward/OS/Users/arind/test_output_24/'
    frame_direc = '/media/edward/OS/Users/arind/new_training_set/'
    # frame_direc = './testing_sample_data'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print("Working with device: ", device)
    
    model_name = 'two_stream_24_semi_supervised'
    start_epoch = 25  # Set this to the epoch you want to continue from
    
    clear_gpu_cache()
    train_model(num_epochs, frame_direc, device, 2, model_name, start_epoch)

if __name__ == "__main__":
    main()