import pandas as pd
from trajectory.preprocess import preprocess_and_sample_by_mmsi
from trajectory.latent_ode_experiment import latent_ode_experiment
from trajectory.resnet_experiment import resnet_experiment
from trajectory.polyreg_experiment import polyreg_experiment
from trajectory.evaluation import evaluate_model_results
from trajectory.visualization import visualize_results_by_observed_points_and_model

if __name__ == "__main__":
    # 데이터 로드 및 전처리
    file_path = "ais.csv"
    data = pd.read_csv(file_path)
    data['positionsourcedate'] = pd.to_datetime(data['positionsourcedate'])
    processed_data = preprocess_and_sample_by_mmsi(data)

    # 실험 실행
    observed_points_list = [10, 30, 50]
    poly_results = polyreg_experiment(processed_data, observed_points_list)
    resnet_results = resnet_experiment(processed_data, observed_points_list)
    latent_ode_results = latent_ode_experiment(processed_data, observed_points_list)

    # 평가
    poly_metrics = evaluate_model_results(poly_results)
    resnet_metrics = evaluate_model_results(resnet_results)
    latent_ode_metrics = evaluate_model_results(latent_ode_results)

    # 결과 출력
    print("Polynomial Regression Metrics:", poly_metrics)
    print("ResNet Metrics:", resnet_metrics)
    print("Latent ODE Metrics:", latent_ode_metrics)

    # 시각화
    visualize_results_by_observed_points_and_model(
        processed_data, poly_results, model_name="PolyReg", observed_points_list=observed_points_list
    )
    visualize_results_by_observed_points_and_model(
        processed_data, resnet_results, model_name="ResNet", observed_points_list=observed_points_list
    )
    visualize_results_by_observed_points_and_model(
        processed_data, latent_ode_results, model_name="Latent ODE", observed_points_list=observed_points_list
    )
