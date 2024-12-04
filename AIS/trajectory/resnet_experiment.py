import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam


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


class SimpleResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super(SimpleResNet, self).__init__()
        self.input_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)


def resnet_experiment(processed_data, hidden_dim=32, epochs=100, lr=0.01, max_plots=12, plots_per_row=4, device="cpu"):
    """
    Train ResNet on processed data with 100 sampled points per MMSI and visualize results.
    """
    num_plots = min(max_plots, len(processed_data))
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()

    for idx in range(num_plots):
        data = processed_data[idx]
        mmsi = data['mmsi']
        timestamps = data['timestamps']
        coordinates = data['coordinates']

        # Normalize coordinates
        mean_coords = np.mean(coordinates, axis=0)
        std_coords = np.std(coordinates, axis=0)
        normalized_coordinates = (coordinates - mean_coords) / std_coords

        # Prepare tensors
        input_dim = 1
        output_dim = 2
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
        coordinates_tensor = torch.tensor(normalized_coordinates, dtype=torch.float32).permute(1, 0).unsqueeze(0).to(device)

        # Define ResNet model
        model = SimpleResNet(input_dim, hidden_dim, output_dim).to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(timestamps_tensor)
            loss = criterion(output, coordinates_tensor)
            loss.backward()
            optimizer.step()

        # Predict on full timestamps
        model.eval()
        with torch.no_grad():
            predicted_trajectory = model(timestamps_tensor).squeeze(0).permute(1, 0).cpu().numpy()
        predicted_trajectory = predicted_trajectory * std_coords + mean_coords

        # Plot results
        ax = axes[idx]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], label='Data Points', color='blue', s=10)
        ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label='ResNet Prediction', color='red', linestyle='--', linewidth=2)
        ax.legend(fontsize=8)
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        ax.set_title(f"MMSI: {mmsi}", fontsize=10)
        ax.grid()

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.show()
