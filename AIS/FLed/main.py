import pandas as pd
from ais_data_utils import preprocess_and_sample_by_mmsi
from models.polynomial_regression import fit_polynomial_regression
from visualization import plot_results

if __name__ == "__main__":
    # 데이터 로드
    file_path = "ais.csv"
    data = pd.read_csv(file_path)
    data['positionsourcedate'] = pd.to_datetime(data['positionsourcedate'])

    # 데이터 전처리
    processed_data = preprocess_and_sample_by_mmsi(data)

    # Polynomial Regression 실행 및 시각화
    poly_predictions = fit_polynomial_regression(processed_data)
    plot_results(processed_data, poly_predictions, labels='Polynomial Regression')
    resnet_experiment(processed_data, hidden_dim=32, epochs=50, lr=0.01, device='cpu')
