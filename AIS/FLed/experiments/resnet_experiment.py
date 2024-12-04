import torch
import numpy as np
from models.resnet import SimpleResNet
import matplotlib.pyplot as plt

def resnet_experiment(processed_data, hidden_dim=32, epochs=100, lr=0.01, max_plots=12, plots_per_row=4, device='cpu'):
    """
    Train ResNet on processed data with 100 sampled points per MMSI and visualize results.
    
    Parameters:
    - processed_data: Preprocessed data with 100-point sampling per MMSI.
    - hidden_dim: Number of hidden units in ResNet.
    - epochs: Number of training epochs.
    - lr: Learning rate.
    - max_plots: Maximum number of plots to display.
    - plots_per_row: Number of plots per row.
    - device: Device to run the experiment (e.g., 'cpu' or 'cuda').
    """
    num_plots = min(max_plots, len(processed_data))
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=plots_per_row, figsize=(18, 12), constrained_layout=True
    )
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        ax.set_title(f"MMSI: {mmsi
