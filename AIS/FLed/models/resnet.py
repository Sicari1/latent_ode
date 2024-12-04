import torch
from torch import nn

class SimpleResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(SimpleResNet, self).__init__()
        self.input_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        # x: [batch, input_dim, seq_length]
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)
