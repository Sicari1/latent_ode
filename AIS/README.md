# Trajectory Modeling and Analysis Framework


# Overview
### This project provides a comprehensive framework for trajectory modeling, prediction, and evaluation using multiple machine learning models, including:
- Polynomial Regression
- ResNet
- Neural ODE
- Latent ODE

### The framework includes modules for data preprocessing, model training, trajectory prediction, and evaluation with various metrics such as MSE, MAE, DTW, and R² scores.

# Features
## 1. Data Preprocessing:
- Trajectory data is preprocessed by sampling points for each unique MMSI.
- Data normalization is applied for consistent processing across models.

## 2. Models:
- Polynomial Regression: Simple polynomial fitting for trajectory prediction.
- ResNet: Deep residual networks for sequence prediction.
- Neural ODE: Continuous-time models for trajectory prediction.
- Latent ODE: Advanced ODE-based latent variable models for irregularly sampled trajectories.

## 3. Evaluation Metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Dynamic Time Warping (DTW)
- Coefficient of Determination (R²)

## 4. Visualization:
- Trajectory data and model predictions are visualized for easy comparison.
- Multiple models and observation levels can be plotted.

# Project Structure
.
├── trajectory
│   ├── preprocess.py             # Data preprocessing and sampling
│   ├── visualization.py          # Trajectory visualization functions
│   ├── polyreg_experiment.py     # Polynomial Regression implementation
│   ├── resnet_experiment.py      # ResNet implementation
│   ├── neural_ode_experiment.py  # Neural ODE implementation
│   ├── latent_ode_experiment.py  # Latent ODE implementation
│   ├── evaluation.py             # Model evaluation metrics
│   └── __init__.py               # Package initializer
├── main.py                       # Entry point for running experiments
└── README.md                     # Project documentation


# Installation
### 1. Clone the repository:
git clone https://github.com/your-repo/trajectory-modeling.git
cd trajectory-modeling

### 2. Install required Python packages:
pip install -r requirements.txt

# Usage
```python
# 1. Preprocess Data
from trajectory.preprocess import preprocess_and_sample_by_mmsi

data = pd.read_csv("ais.csv")
data['positionsourcedate'] = pd.to_datetime(data['positionsourcedate'])
processed_data = preprocess_and_sample_by_mmsi(data)

# 2. Run Experiments
from trajectory.polyreg_experiment import polyreg_experiment
from trajectory.resnet_experiment import resnet_experiment
from trajectory.latent_ode_experiment import latent_ode_experiment

observed_points_list = [10, 30, 50]
poly_results = polyreg_experiment(processed_data, observed_points_list)
resnet_results = resnet_experiment(processed_data, observed_points_list)
latent_ode_results = latent_ode_experiment(processed_data, observed_points_list)

# 3. Evaluate Results
from trajectory.evaluation import evaluate_model_results

poly_metrics = evaluate_model_results(poly_results)
resnet_metrics = evaluate_model_results(resnet_results)
latent_ode_metrics = evaluate_model_results(latent_ode_results)

print("Polynomial Regression Metrics:", poly_metrics)
print("ResNet Metrics:", resnet_metrics)
print("Latent ODE Metrics:", latent_ode_metrics)

# 4. Visualize Results
from trajectory.visualization import visualize_results_by_observed_points_and_model

visualize_results_by_observed_points_and_model(
    processed_data, poly_results, model_name="PolyReg", observed_points_list=[10, 30, 50]
)
```

# Example Workflow
1. Load and preprocess the AIS trajectory data. (Sorry for not giving my AIS data since it is project-private data)
2. Run experiments with different models.
3. Evaluate the performance using multiple metrics.
4. Visualize the predicted trajectories and compare model outputs.


# Evaluation Metrics
The project uses the following metrics to evaluate predicted trajectories against ground truth:
- MSE (Mean Squared Error): Measures the average squared difference between predicted and actual coordinates.
- MAE (Mean Absolute Error): Measures the average absolute difference.
- DTW (Dynamic Time Warping): Measures the similarity between two trajectories.
- R² (Coefficient of Determination): Indicates how well the predictions match the actual values.


# Dependencies
- Python 3.8+
- Required libraries:
    - numpy
    - pandas
    - matplotlib
    - scikit-learn
    - torch
    - torchdiffeq
    - fastdtw
### Install dependencies with:
pip install -r requirements.txt


# Future Work
- Implement Physics-informed NN and Physics-informed ODE and Neural Flow models for comparison.
- Explore hyperparameter optimization for Neural ODE and Latent ODE models.

# Contributors
- Seongjin Kim - https://github.com/Sicari1


