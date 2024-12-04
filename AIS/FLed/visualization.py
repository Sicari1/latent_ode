import matplotlib.pyplot as plt

def plot_results(processed_data, predictions, labels, max_plots=12, plots_per_row=4):
    """
    Visualize predictions alongside the original data.
    """
    num_plots = min(max_plots, len(processed_data))
    num_rows = (num_plots + plots_per_row - 1) // plots_per_row
    fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(18, 12), constrained_layout=True)
    axes = axes.flatten()

    for idx in range(num_plots):
        data = processed_data[idx]
        mmsi = data['mmsi']
        coordinates = data['coordinates']
        prediction = predictions[idx]
        ax = axes[idx]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], label='Original Data', s=10, color='blue')
        ax.plot(prediction[:, 0], prediction[:, 1], label=f'{labels} Prediction', linestyle='--', color='red', linewidth=2)
        ax.set_title(f"MMSI: {mmsi}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid()

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    plt.show()
