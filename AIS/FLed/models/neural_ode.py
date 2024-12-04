import torch
from torch import nn

class ODEF(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        return self.net(x)


class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, z0, t):
        trajectory = [z0]
        z = z0
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            z = z + dt * self.func(z, t[i])
            trajectory.append(z)
        return torch.stack(trajectory)
