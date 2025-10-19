import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    filters: int = 32
    units1: int = 128
    units2: int = 64
    dropout_dense_rate: float = 0.2
    dropout_conv_rate: float = 0.0
    use_batchnorm: bool = True
    use_maxpooling: bool =True
    num_classes: int = 2
    num_blocks: int = 3

class CNN(nn.Module):
    def __init__(self, config: ModelConfig,  input_size=(16, 3, 128, 128)):
        super().__init__()
        self.in_channels = input_size[1]
        self.filters = config.filters
        self.units1 = config.units1
        self.units2 = config.units2
        self.dropout_dense_rate = config.dropout_dense_rate
        self.dropout_conv_rate = config.dropout_conv_rate
        self.use_batchnorm = config.use_batchnorm
        self.use_maxpool = config.use_maxpooling
        self.num_classes = config.num_classes
        self.num_blocks = config.num_blocks
        in_ch = self.in_channels

        self.conv_blocks = nn.ModuleList()
        for _ in range(self.num_blocks):
            layers = []
            layers.append(nn.Conv2d(in_ch, self.filters, kernel_size=3, padding=1))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm2d(self.filters))
            layers.append(nn.ReLU())
            if self.use_maxpool:
                layers.append(nn.MaxPool2d(2))
            if self.dropout_conv_rate > 0:
                layers.append(nn.Dropout2d(self.dropout_conv_rate))
            self.conv_blocks.append(nn.Sequential(*layers))
            in_ch = self.filters 

        activation_map_size = self._conv_test(input_size)
        self.agg = nn.AvgPool2d(activation_map_size)

        dense_layers = [
            nn.Flatten(),
            nn.Linear(self.filters, self.units1),
            nn.ReLU(),
            nn.Linear(self.units1, self.units2),
            nn.ReLU(),
        ]

        if self.dropout_dense_rate > 0:
            dense_layers.append(nn.Dropout(self.dropout_dense_rate))
        
        dense_layers.append(nn.Linear(self.units2, self.num_classes))

        self.dense = nn.Sequential(*dense_layers)
    
    def _conv_test(self, input_size):
        with torch.no_grad():
            x = torch.ones(input_size)
            for block in self.conv_blocks:
                x = block(x)
            return x.shape[-2:]

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.agg(x)
        return self.dense(x)