import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import Tensor
from torchvision.models import transformer

class TimeSeriesDetrEncoder(nn.Module):
    class TimeSeriesDetrEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super(TimeSeriesDetrEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.input_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.position_embedding = nn.Parameter(torch.randn(1, hidden_dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x.permute(0, 2, 1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.input_proj(x)
        x = x.permute(2, 0, 1)  # (T, N, C)
        x = x + self.position_embedding
        memory = self.transformer_encoder(x)
        return memory

class TimeSeriesDetrDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers:int, num_classes: int, num_angles: int, num_heads: int, dropout: float):
        super(TimeSeriesDetrDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_angles = num_angles
        self.class_output = nn.Linear(hidden_dim, num_classes)
        self.angle_output = nn.Linear(hidden_dim, num_angles)
        self.attribute_output = nn.Linear(hidden_dim, 1)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory: Tensor, object_queries: Tensor) -> List[Tensor]:
        tgt = object_queries.unsqueeze(0)  # (1, N, C)
        decoder_output = self.transformer_decoder(tgt, memory)  # (1, N, C)
        decoder_output = decoder_output.squeeze(0)  # (N, C)
        class_logits = self.class_output(decoder_output)
        angle_logits = self.angle_output(decoder_output)
        att_logits = self.attribute_output(decoder_output)
        return [class_logits, angle_logits], att_logits
    
class TimeSeriesDETR(nn.Module):
    def __init__(self, input_dim, num_classes, num_queries, kernel_size=4,
                  stride=4, num_layers=6, num_heads=8, dropout=0.1):
        super(TimeSeriesDETR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=input_dim, kernel_size=kernel_size, padding='same', stride=1)
        self.pool = nn.MaxPool2d(kernel_size=stride)
        self.skip = nn.Conv2d(in_channels=1, out_channels=input_dim, kernel_size=1, padding=0, stride=stride)
        self.drop = nn.Dropout2d(p=dropout)
        self.batchnorm1 = nn.BatchNorm2d(input_dim)   
        self.backbone = nn.Sequential(
            [self.conv1, self.pool, self.skip, self.drop, self.batchnorm1]) 
        self.transfomer_layer = transformer.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = transformer.TransformerEncoder(self.transfomer_layer, num_layers=num_layers)
        self.query_embed = nn.Embedding(num_queries, input_dim)
        self.linear_class = nn.Linear(input_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(input_dim, 4)

    def forward(self, x):
        # Assuming x has shape [batch_size, seq_len, input_dim]
        # Pass input through transformer encoder
        x = self.backbone(x)
        x = self.transformer(x)
        
        # Generate queries
        queries = self.query_embed(torch.arange(x.size(0)))
        
        # Concatenate queries with transformer outputs
        x = torch.cat([queries.unsqueeze(1).repeat(1, x.size(1), 1), x], dim=-1)
        
        # Predict class scores
        class_scores = self.linear_class(x)
        
        # Predict bounding box coordinates
        bbox_preds = self.linear_bbox(x)
        
        return class_scores, bbox_preds

class TimeSeriesDetrModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float, num_classes: int, num_angles: int, num_queries: int):
        super(TimeSeriesDetrModel, self).__init__()
        self.encoder = TimeSeriesDetrEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.decoder = TimeSeriesDetrDecoder(hidden_dim, num_classes, num_angles, num_heads, dropout)
        self.object_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

    def forward(self, x: Tensor) -> List[Tensor]:
        memory = self.encoder(x)
        return self.decoder(memory, self.object_queries)
    



