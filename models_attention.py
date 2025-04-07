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
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 32 * 32, 128),
                nn.LayerNorm(128),
                nn.ReLU()
            ) for _ in range(4)  # One processor per feature
        ])

    def forward(self, x):
        # Input: [B, 4, 128, 128, 24]
        batch_size = x.shape[0]
        
        # Process each feature independently
        processed_features = []
        for i in range(4):  # For each feature
            # Get all frames for this feature [B,128,128,24]
            feature = x[:, i]
            # Average across time dimension [B,128,128]
            feature = feature.mean(dim=2)
            # Add channel dimension [B,1,128,128]
            feature = feature.unsqueeze(1)
            # Process through feature-specific CNN
            processed = self.feature_processors[i](feature)
            processed_features.append(processed)
        
        # Combine features [B,4,128]
        x = torch.stack(processed_features, dim=1)
        # Average across features [B,128]
        return x.mean(dim=1)

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
            nn.Linear(256, 64),  # Reduced capacity
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, frames, vaf_features):
        # Process VAF Features [B,24,4,128,128] -> [B,4,128,128,24]
        vaf_processed = self.vaf_processor(vaf_features.permute(0, 2, 3, 1, 4))
        
        # Temporal stream processing
        temporal_features = self.temporal_stream(frames.permute(0, 2, 1, 3, 4))
        
        # Cross-Attention
        attended_features, _ = self.cross_attention(
            query=vaf_processed.unsqueeze(1),  # [B,1,128]
            key=temporal_features.unsqueeze(1),  # [B,1,128]
            value=temporal_features.unsqueeze(1)  # [B,1,128]
        )
        
        # Combine VAF + Attended features
        combined = torch.cat([
            vaf_processed,  # Original VAF [B,128]
            attended_features.squeeze(1)  # Attended [B,128]
        ], dim=1)  # [B,256]
        
        return self.prediction_head(combined)