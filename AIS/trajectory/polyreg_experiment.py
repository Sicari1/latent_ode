from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

def polyreg_experiment(processed_data, observed_points_list, degree=3):
    """
    Conducts polynomial regression experiments on processed data.
    """
    results = []
    for observed_points in observed_points_list:
        for data in processed_data:
            timestamps = data['timestamps']
            coordinates = data['coordinates']

            observed_indices = np.linspace(0, len(timestamps) - 1, observed_points, dtype=int)
            observed_timestamps = timestamps[observed_indices]
            observed_coordinates = coordinates[observed_indices]

            mean_coords = np.mean(coordinates, axis=0)
            std_coords = np.std(coordinates, axis=0)
            normalized_observed = (observed_coordinates - mean_coords) / std_coords

            poly = PolynomialFeatures(degree=degree)
            observed_poly = poly.fit_transform(observed_timestamps.reshape(-1, 1))

            longitude_model = LinearRegression()
            latitude_model = LinearRegression()
            longitude_model.fit(observed_poly, normalized_observed[:, 0])
            latitude_model.fit(observed_poly, normalized_observed[:, 1])

            full_poly = poly.transform(timestamps.reshape(-1, 1))
            predicted_longitude = longitude_model.predict(full_poly)
            predicted_latitude = latitude_model.predict(full_poly)

            predicted_trajectory = np.stack([predicted_longitude, predicted_latitude], axis=1)
            predicted_trajectory = predicted_trajectory * std_coords + mean_coords

            results.append({
                'observed_points': observed_points,
                'mmsi': data['mmsi'],
                'full_coordinates': coordinates,
                'predicted_trajectory': predicted_trajectory
            })

    return results
