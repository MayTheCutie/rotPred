import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor
import math
# from torchvision.models import transformer


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
        self.positional_encoding = PositionalEncoding(hidden_dim)

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
        # Add positional encoding
        x = self.positional_encoding(x)
        memory = self.transformer_encoder(x)
        return memory

class TimeSeriesDetrDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers:int, 
      num_heads: int, dropout: float,):
        super(TimeSeriesDetrDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, memory: Tensor, tgt: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        t, bs, h = tgt.shape
        hs = self.transformer_decoder(tgt, memory)  # (1, N, C)
        return hs
    
class TimeSeriesDetrModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int,
     dropout: float, num_classes: int, num_angles: int, num_queries: int, mlp_layers: int = 2, num_atts: int = 2):
        super(TimeSeriesDetrModel, self).__init__()
        self.encoder = TimeSeriesDetrEncoder(input_dim, hidden_dim, num_layers, num_heads, dropout)
        self.decoder = TimeSeriesDetrDecoder(hidden_dim, num_layers, num_heads, dropout)
        self.num_classes = num_classes
        self.num_angles = num_angles
        self.object_queries = nn.Embedding(num_queries, hidden_dim)
        self.class_output = nn.Linear(hidden_dim, num_classes) # spot classification
        self.bbox_output = MLP(hidden_dim,hidden_dim, num_angles, mlp_layers) # spot angles regression
        self.attribute_output = MLP(hidden_dim, hidden_dim//4, num_atts, mlp_layers) # global attributes (inclination)

    def forward(self, x: Tensor) -> List[Tensor]:
        if x.dim() == 3 and x.size(1) == 1:
            x = x.transpose(2, 1)
        bs, t, c = x.shape
        memory = self.encoder(x)
        query_embed = self.object_queries.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(memory, tgt)
        decoder_output = hs.transpose(0, 1)
        class_logits = self.class_output(decoder_output) 
        bbox_logits = self.bbox_output(decoder_output).sigmoid()
        att_logits = self.attribute_output(decoder_output.sum(dim=1))
        out_dict = {'pred_boxes': bbox_logits, 'pred_logits': class_logits}
        return out_dict, att_logits
    



