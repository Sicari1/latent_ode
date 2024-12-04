import numpy as np

def preprocess_and_sample_by_mmsi(data, min_points=100):
    """
    Processes and samples data by MMSI, ensuring each trajectory has at least `min_points` points.
    """
    processed_data = []
    unique_mmsis = data['mmsi'].unique()

    for mmsi in unique_mmsis:
        mmsi_data = data[data['mmsi'] == mmsi].sort_values('positionsourcedate')

        if len(mmsi_data) < min_points:
            continue

        timestamps = (mmsi_data['positionsourcedate'] - mmsi_data['positionsourcedate'].iloc[0]).dt.total_seconds()
        coordinates = mmsi_data[['longitude', 'latitude']].values

        sampled_timestamps = np.linspace(0, timestamps.max(), min_points)
        sampled_longitudes = np.interp(sampled_timestamps, timestamps, coordinates[:, 0])
        sampled_latitudes = np.interp(sampled_timestamps, timestamps, coordinates[:, 1])

        processed_data.append({
            'mmsi': mmsi,
            'timestamps': sampled_timestamps / timestamps.max(),
            'coordinates': np.stack([sampled_longitudes, sampled_latitudes], axis=1)
        })

    return processed_data
