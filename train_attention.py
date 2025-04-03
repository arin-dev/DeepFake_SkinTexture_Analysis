import torch
import torch.optim as optim
import torch.nn as nn
from models_attention import TwoStreamNetworkTransferLearning
from data_loader_new_train import get_data_loaders
# from vaf_ext.py import 
import json
import os
import time

def get_device_for_model():
    # Check memory usage on all available GPUs
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(i)
        mem_total = torch.cuda.get_device_properties(i).total_memory
        if mem_alloc < mem_total * 0.95:
            return torch.device(f'cuda:{i}')
    raise MemoryError("All GPUs are out of memory!")

def train_model(num_epochs, frame_direc, device, batch_size=1, model_name=None, start_epoch=0):
    print("Entering to train data!")
    model = TwoStreamNetworkTransferLearning()

    # Use DataParallel with all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    device = get_device_for_model()
    model = model.to(device)
    
    # Load pre-trained model if path is provided
    checkpoint_path = f'models/{model_name}_model_epoch_{start_epoch}.pth'
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
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
    with open('labels.json', 'r') as file:
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

            print(video_names)
            print(type(data), data.shape)
            print(type(vaf_features), len(vaf_features), len(vaf_features[0]))

            data = data.to(device)
            vaf_features = vaf_features.to(device)
            outputs = model(data).squeeze(1)
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
            
            if (batch_count + 1) % 20 == 0:
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
    num_epochs = 1
    # frame_direc = '/kaggle/input/train-output-24-reduced/train_output_24_reduced'
    # frame_direc = '/media/edward/OS/Users/arind/test_output_24/'
    frame_direc = './testing_sample_data'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print("Working with device: ", device)
    
    model_name = 'two_stream_24'
    start_epoch = 0  # Set this to the epoch you want to continue from
    
    clear_gpu_cache()
    train_model(num_epochs, frame_direc, device, 1, model_name, start_epoch)

if __name__ == "__main__":
    main()