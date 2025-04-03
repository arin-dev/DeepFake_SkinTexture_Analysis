import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class TemporalStream3DResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=128, frames_per_clip=12, height=128, width=128):
        super(TemporalStream3DResNet, self).__init__()
        self.resnet3d = models.video.r3d_18(weights='DEFAULT')
        self.resnet3d.stem[0] = nn.Conv3d(input_channels, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3))
        self.resnet3d = nn.Sequential(*list(self.resnet3d.children())[:-1])

        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, frames_per_clip, height, width)
            feature_map = self.resnet3d(dummy_input)

        self.fc = nn.Linear(feature_map.numel(), num_classes)

    def forward(self, x):
        x = self.resnet3d(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class TwoStreamNetworkTransferLearning(nn.Module):
    def __init__(self):
        super(TwoStreamNetworkTransferLearning, self).__init__()
        self.temporal_stream = TemporalStream3DResNet()
        
        # Simplified attention mechanism that works directly on frame features
        self.temporal_attention = nn.Sequential(
            nn.Linear(24*3*128*128, 256),  # Process flattened frames directly
            nn.Tanh(),
            nn.Linear(256, 24),
            nn.Softmax(dim=1)
        )
        # Final layer for temporal features only
        self.fc_final = nn.Linear(128, 1)

    def forward(self, frames):
        batch_size, frames_no, channels, height, width = frames.shape

        # Compute attention weights directly from frames
        flattened_frames = frames[:, 12*2:, :, :, :].reshape(batch_size, -1)
        attention_weights = self.temporal_attention(flattened_frames)
        
        # Apply attention to temporal frames
        temporal_frames = frames[:, 12*2:, :, :, :]
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        temporal_frames = temporal_frames * (1 + attention_weights)
        
        # Process temporal stream
        temporal_input = temporal_frames.permute(0, 2, 1, 3, 4)
        temporal_features = self.temporal_stream(temporal_input)
        
        # Final output
        output = torch.sigmoid(self.fc_final(temporal_features))
        return output