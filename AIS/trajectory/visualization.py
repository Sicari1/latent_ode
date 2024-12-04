import matplotlib.pyplot as plt

def visualize_trajectories(processed_data, model_results=None, max_plots=12, plots_per_row=4, scatter_only=False):
    """
    Visualizes original and predicted trajectories.
    """
    num_plots = min(max_plots, len(processed_data))
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(16, 4 * num_rows), constrained_layout=True)
    axes = axes.flatten()

    for idx in range(num_plots):
        data = processed_data[idx]
        mmsi = data['mmsi']
        coordinates = data['coordinates']

        ax = axes[idx]

        if scatter_only:
            ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=10, label='Original Data')
        else:
            ax.plot(coordinates[:, 0], coordinates[:, 1], c='blue', linestyle='--', label='Original Path')
            ax.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=10)

        if model_results:
            predicted_trajectory = model_results[idx]['predicted_trajectory']
            ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], c='red', label='Predicted Path')

        ax.set_title(f"MMSI: {mmsi}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.grid()

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.show()
