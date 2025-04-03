import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class SpatialStreamResNet(nn.Module):
    def __init__(self):
        super(SpatialStreamResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 128)
    
    def forward(self, x):
        x = self.resnet_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
        self.spatial_stream = SpatialStreamResNet()
        self.temporal_stream = TemporalStream3DResNet()
        
        # Attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 24),
            nn.Softmax(dim=1)
        )
        # Final layer for temporal features only
        self.fc_final = nn.Linear(128, 1)

    def forward(self, frames):
        batch_size, frames_no, channels, height, width = frames.shape

        # Process spatial stream
        spatial_frames = frames[:, :12*2, :, :, :]
        spatial_frames = spatial_frames.reshape(-1, channels, height, width)
        spatial_frames = spatial_frames.permute(0, 2, 3, 1).contiguous()
        spatial_frames = spatial_frames.cpu().numpy()

        processed_spatial_frames = []
        for frame in spatial_frames:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
            processed_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2RGB)
            processed_spatial_frames.append(processed_image)

        processed_spatial_frames = torch.from_numpy(np.array(processed_spatial_frames)).permute(0, 3, 1, 2).to(frames.device)
        spatial_features = self.spatial_stream(processed_spatial_frames)
        spatial_features = spatial_features.reshape(batch_size, 12, -1).mean(dim=1)

        # Compute attention weights using spatial features
        attention_weights = self.temporal_attention(spatial_features)
        
        # Apply attention to temporal frames
        temporal_frames = frames[:, 12*2:, :, :, :]
        
        # Reshape attention weights to match temporal frames
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Add 1 to attention weights to prevent feature reduction
        temporal_frames = temporal_frames * (1 + attention_weights)
        
        # Process temporal stream
        temporal_input = temporal_frames.permute(0, 2, 1, 3, 4)
        temporal_features = self.temporal_stream(temporal_input)
        
        # Final output
        output = torch.sigmoid(self.fc_final(temporal_features))
        return output