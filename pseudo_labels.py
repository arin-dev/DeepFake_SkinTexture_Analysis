import torch
import torch.nn as nn
from models_attention import TwoStreamNetworkTransferLearning
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
    model = model.to(device)
    
    checkpoint_path = 'models/two_stream_24_semi_supervised_model_epoch_25.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        print("No model found")
        return
    
    model.eval()
    results = {}
    correct = 0
    total = 0
    confusion_matrix = {'1_as_1': 0, '0_as_0': 0, '0_as_1': 0, '1_as_0': 0}

    with open('train_real_new.json', 'r') as file:
        label_map = json.load(file)

    test_loader = get_test_loaders(frame_direc, batch_size)
    processed_count = 0
    total_subfolders = len(os.listdir(frame_direc))
    
    with torch.no_grad():
        for data, video_names, vaf_features in test_loader:
            if data is None:
                continue
                
            data = data.float().to(device)
            vaf_features = vaf_features.float().to(device)
            
            outputs = model(data, vaf_features).squeeze(1)
            predicted = (outputs >= threshold).float()
            
            labels_tensor = torch.tensor([
                0 if label_map.get(video_name) == -1 else label_map.get(video_name)
                for video_name in video_names
            ]).float().to(device)
            
            total += labels_tensor.size(0)
            correct += (predicted == labels_tensor).sum().item()

            for video_name, pred, label in zip(video_names, predicted, labels_tensor):
                results[video_name] = int(pred.item())
                
                # Update confusion matrix
                if label == 1 and pred == 1:
                    confusion_matrix['1_as_1'] += 1
                elif label == 0 and pred == 0:
                    confusion_matrix['0_as_0'] += 1
                elif label == 0 and pred == 1:
                    confusion_matrix['0_as_1'] += 1
                elif label == 1 and pred == 0:
                    confusion_matrix['1_as_0'] += 1

            processed_count += len(video_names)
            if processed_count % 20 == 0:
                remaining = total_subfolders - processed_count
                print(f"Processed {processed_count} subfolders, {remaining} remaining")

    with open('model_outputs.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved model outputs to model_outputs.json")

    # Print accuracy and confusion matrix
    accuracy = correct / total * 100
    print(f'Final Accuracy: {accuracy:.2f}% (Threshold: {threshold})')
    print('Confusion Matrix:')
    print(f"1 correctly detected as 1: {confusion_matrix['1_as_1']}")
    print(f"0 correctly detected as 0: {confusion_matrix['0_as_0']}")
    print(f"0 incorrectly detected as 1: {confusion_matrix['0_as_1']}")
    print(f"1 incorrectly detected as 0: {confusion_matrix['1_as_0']}")

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
    # reset_gpu()
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary classification')
    args = parser.parse_args()

    # frame_direc = '/media/edward/OS/Users/arind/test_output_24'
    frame_direc = '/media/edward/OS/Users/arind/train_output_24/'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    test_model(frame_direc, device, 30, args.threshold)
