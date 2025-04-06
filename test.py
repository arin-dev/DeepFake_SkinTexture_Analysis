import torch
import torch.nn as nn
from models_attention import TwoStreamNetworkTransferLearning
# from data_loader_test import get_test_loaders
from data_loader_test import get_test_loaders
import json
import os
import argparse

def get_device_for_model():
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        mem_alloc = torch.cuda.memory_allocated(i)
        mem_total = torch.cuda.get_device_properties(i).total_memory
        if mem_alloc < mem_total * 0.95:
            return torch.device(f'cuda:{i}')
    raise MemoryError("All GPUs are out of memory!")

def test_model(frame_direc, device, batch_size, threshold=0.5):
    model = TwoStreamNetworkTransferLearning()
    
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    # device = get_device_for_model()
    model = model.to(device)
    
    checkpoint_path = 'two_stream_24_model_epoch_15.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        state_dict = checkpoint['model_state_dict']
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    model.eval()

    print("Opening Label.json!")
    # with open('real_labels.json', 'r') as file:
    with open('real_labels_not_used.json', 'r') as file:
        label_map = json.load(file)

    correct = 0
    total = 0

    test_loader = get_test_loaders(frame_direc, batch_size)
    with torch.no_grad():
        for data, video_names, vaf_features in test_loader:
            if data is None:
                continue
                
            print(f"Processing batch: {video_names}")
            data = data[:, 12*2:, :, :, :].float().to(device)
            vaf_features = vaf_features.float().to(device)
            
            outputs = model(data, vaf_features).squeeze(1)
            
            labels_tensor = torch.tensor([
                0 if label_map.get(video_name) == -1 else label_map.get(video_name)
                for video_name in video_names
            ]).float().to(device)
            
            predicted = (outputs >= threshold).float()
            total += labels_tensor.size(0)
            correct += (predicted == labels_tensor).sum().item()

            print(f"Model outputs: {outputs.cpu().numpy()}")
            print(f"Predictions: {predicted.cpu().numpy()}")
            print(f"Ground truth: {labels_tensor.cpu().numpy()}")

    accuracy = correct / total * 100
    print(f'Final Accuracy: {accuracy:.2f}% (Threshold: {threshold})')

def reset_gpu():
    import gc
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        torch.cuda.ipc_collect()
        print("GPU fully reset")
    else:
        print("No GPU available")

if __name__ == "__main__":
    reset_gpu()
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    args = parser.parse_args()

    # frame_direc = '/media/edward/OS/Users/arind/test_output_24'
    frame_direc = '/media/edward/OS/Users/arind/train_output_24/not_training_on_this'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_model(frame_direc, device, 1, args.threshold)
