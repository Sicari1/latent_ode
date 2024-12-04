from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def fit_polynomial_regression(data, degree=3):
    """
    Fit and predict trajectories using Polynomial Regression.
    """
    predictions = []
    for item in data:
        timestamps = item['timestamps'].reshape(-1, 1)
        coordinates = item['coordinates']
        poly = PolynomialFeatures(degree=degree)
        timestamps_poly = poly.fit_transform(timestamps)
        longitude_model = LinearRegression().fit(timestamps_poly, coordinates[:, 0])
        latitude_model = LinearRegression().fit(timestamps_poly, coordinates[:, 1])
        predicted_longitude = longitude_model.predict(timestamps_poly)
        predicted_latitude = latitude_model.predict(timestamps_poly)
        predictions.append(np.stack([predicted_longitude, predicted_latitude], axis=1))
    return predictions
