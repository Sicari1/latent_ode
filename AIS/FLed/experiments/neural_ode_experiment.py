import torch
import numpy as np
from models.neural_ode import ODEF, NeuralODE
import matplotlib.pyplot as plt

def neural_ode_prediction(processed_data, max_plots=12, plots_per_row=4, device='cpu'):
    """
    Perform Neural ODE prediction on processed data (100-point sampled).
    
    Parameters:
    - processed_data: Preprocessed data with 100-point sampling per MMSI.
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
        coordinates = data['coordinates']
        timestamps = data['timestamps']

        # Normalize coordinates
        mean_coords = np.mean(coordinates, axis=0)
        std_coords = np.std(coordinates, axis=0)
        normalized_coords = (coordinates - mean_coords) / std_coords

        coordinates_tensor = torch.tensor(normalized_coords, dtype=torch.float32).to(device)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32).to(device)

        # Define Neural ODE
        input_dim = 2
        hidden_dim = 32
        ode_func = ODEF(input_dim, hidden_dim).to(device)
        model = NeuralODE(ode_func).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training
        epochs = 1000
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

        # Plot
        ax = axes[idx]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], label='Data Points', color='blue', s=10)
        ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label='Latent ODE Prediction', color='red', linestyle='--', linewidth=2)
        ax.legend(fontsize=8)
        ax.set_xlabel('Longitude', fontsize=8)
        ax.set_ylabel('Latitude', fontsize=8)
        ax.set_title(f"MMSI: {mmsi}", fontsize=10)
        ax.grid()

    # Remove unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.show()
