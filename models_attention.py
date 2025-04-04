import torch
import torch.nn as nn
import torchvision.models as models

class TemporalStream3DResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=128, frames_per_clip=24, height=128, width=128):
        super(TemporalStream3DResNet, self).__init__()
        self.resnet3d = models.video.r3d_18(weights='DEFAULT')
        self.resnet3d.stem[0] = nn.Conv3d(
            input_channels, 64, 
            kernel_size=(7, 7, 7), 
            stride=(1, 2, 2), 
            padding=(3, 3, 3)
        )
        self.resnet3d = nn.Sequential(*list(self.resnet3d.children())[:-1])

        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, frames_per_clip, height, width)
            feature_map = self.resnet3d(dummy_input)

        self.fc = nn.Linear(feature_map.numel(), num_classes)

    def forward(self, x):
        x = self.resnet3d(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class VAFProcessor(nn.Module):
    def __init__(self):
        super(VAFProcessor, self).__init__()
        self.conv3d = nn.Sequential(
            # Process each feature separately
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 6 * 32 * 32, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

    def forward(self, x):
        # Input: [B, 4, 128, 128, 3, 24]
        batch_size = x.shape[0]
        
        # Reshape to process each feature separately
        x = x.permute(0, 1, 4, 5, 2, 3)  # [B,4,3,24,128,128]
        x = x.reshape(-1, 1, 24, 128, 128)  # [B*4*3,1,24,128,128]
        
        # Process through conv3d
        x = self.conv3d(x)
        
        # Reshape back and average across features and channels
        x = x.view(batch_size, 4, 3, -1)  # [B,4,3,128]
        x = x.mean(dim=[1,2])  # [B,128]
        return x

class TwoStreamNetworkTransferLearning(nn.Module):
    def __init__(self):
        super(TwoStreamNetworkTransferLearning, self).__init__()
        self.temporal_stream = TemporalStream3DResNet(frames_per_clip=24)
        self.vaf_processor = VAFProcessor()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=4,
            batch_first=True
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, frames, vaf_features):
        # Process VAF Features [B,4,128,128,3,24]
        vaf_processed = self.vaf_processor(vaf_features)
        
        # Process Temporal Stream [B,C,T,H,W]
        temporal_input = frames.permute(0, 2, 1, 3, 4)
        temporal_features = self.temporal_stream(temporal_input)
        
        # Cross-Modal Attention
        attended_features, _ = self.cross_attention(
            query=vaf_processed.unsqueeze(1),
            key=temporal_features.unsqueeze(1),
            value=temporal_features.unsqueeze(1)
        )
        
        # Combine Features
        combined = torch.cat([
            temporal_features,
            attended_features.squeeze(1)
        ], dim=1)
        
        return self.prediction_head(combined)