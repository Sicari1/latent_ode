import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


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


def neural_ode_experiment(processed_data, hidden_dim=32, epochs=1000, lr=0.01, max_plots=12, plots_per_row=4, device="cpu"):
    """
    Perform Neural ODE prediction on processed data.
    """
    num_plots = min(max_plots, len(processed_data))
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()

    for idx in range(num_plots):
        data = processed_data[idx]
        mmsi = data['mmsi']
        coordinates = data['coordinates']
        timestamps = data['timestamps']

        # Normalize coordinates
        mean_coords = np.mean(coordinates, axis=0)
        std_coords = np.std(coordinates, axis=0)
        normalized_coordinates = (coordinates - mean_coords) / std_coords

        coordinates_tensor = torch.tensor(normalized_coordinates, dtype=torch.float32).to(device)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32).to(device)

        # Define Neural ODE
        input_dim = 2
        ode_func = ODEF(input_dim, hidden_dim).to(device)
        model = NeuralODE(ode_func).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training
        for epoch in range(epochs):
            optimizer.zero_grad()
            z0 = coordinates_tensor[0]
            t = timestamps_tensor
            z_pred = model(z0, t)
            loss = criterion(z_pred, coordinates_tensor)
            loss.backward()
            optimizer.step()

        # Prediction
        z0 = coordinates_tensor[0]
        predicted_trajectory = model(z0, timestamps_tensor).detach().cpu().numpy()
        predicted_trajectory = predicted_trajectory * std_coords + mean_coords

        # Plot results
        ax = axes[idx]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], label='Data Points', color='blue', s=10)
        ax.plot(predicted_
