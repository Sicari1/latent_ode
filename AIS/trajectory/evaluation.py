import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def evaluate_trajectory(full_coordinates, predicted_trajectory):
    """
    Evaluates predicted trajectory against ground truth using multiple metrics.
    
    Parameters:
    - full_coordinates: Ground truth trajectory (Nx2 array).
    - predicted_trajectory: Predicted trajectory (Nx2 array).
    
    Returns:
    - metrics: Dictionary containing MSE, MAE, DTW, and R^2 scores.
    """
    min_len = min(len(full_coordinates), len(predicted_trajectory))
    full_coordinates = full_coordinates[:min_len]
    predicted_trajectory = predicted_trajectory[:min_len]

    mse = mean_squared_error(full_coordinates, predicted_trajectory)
    mae = mean_absolute_error(full_coordinates, predicted_trajectory)
    r2 = r2_score(full_coordinates, predicted_trajectory)

    # Compute DTW distance
    dtw_distance, _ = fastdtw(full_coordinates, predicted_trajectory, dist=euclidean)

    return {
        'MSE': mse,
        'MAE': mae,
        'DTW': dtw_distance,
        'R2': r2
    }

def evaluate_model_results(results):
    """
    Evaluates all model results and returns average metrics.
    
    Parameters:
    - results: List of dictionaries with 'full_coordinates' and 'predicted_trajectory'.
    
    Returns:
    - avg_metrics: Dictionary with averaged metrics across all results.
    """
    metrics_list = [evaluate_trajectory(r['full_coordinates'], r['predicted_trajectory']) for r in results]
    
    avg_metrics = {
        'MSE': np.mean([m['MSE'] for m in metrics_list]),
        'MAE': np.mean([m['MAE'] for m in metrics_list]),
        'DTW': np.mean([m['DTW'] for m in metrics_list]),
        'R2': np.mean([m['R2'] for m in metrics_list]),
    }
    return avg_metrics
