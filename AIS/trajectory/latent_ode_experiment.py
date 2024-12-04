import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from trajectory.visualization import visualize_results_by_observed_points_and_model

def sample_standard_gaussian(mu, sigma):
    device = mu.device
    d = torch.distributions.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()

def split_last_dim(data):
    last_dim = data.size()[-1] // 2
    return data[..., :last_dim], data[..., last_dim:]

def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, device=torch.device("cpu")):
        super(Encoder_z0_ODE_RNN, self).__init__()
        self.gru = nn.GRU(input_dim, latent_dim, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(latent_dim, 50), nn.Tanh(), nn.Linear(50, latent_dim * 2)
        )
        init_network_weights(self.hiddens_to_z0)

    def forward(self, data):
        outputs, _ = self.gru(data)
        last_output = outputs[:, -1]
        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        return mean, std.abs()

class ODEFunc(nn.Module):
    def __init__(self, latent_dim):
        super(ODEFunc, self).__init__()
        self.gradient_net = nn.Sequential(
            nn.Linear(latent_dim, 50), nn.Tanh(), nn.Linear(50, latent_dim)
        )
        init_network_weights(self.gradient_net)

    def forward(self, t, y):
        return self.gradient_net(y)

class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method="rk4", device=torch.device("cpu")):
        super(DiffeqSolver, self).__init__()
        self.ode_func = ode_func
        self.method = method
        self.device = device

    def forward(self, z0, time_points):
        return odeint(self.ode_func, z0, time_points, method=self.method)

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(latent_dim, input_dim)
        init_network_weights(self.decoder)

    def forward(self, latent):
        return self.decoder(latent)

class LatentODESimple(nn.Module):
    def __init__(self, input_dim, latent_dim, device=torch.device("cpu")):
        super(LatentODESimple, self).__init__()
        self.encoder = Encoder_z0_ODE_RNN(latent_dim, input_dim, device=device)
        self.ode_func = ODEFunc(latent_dim)
        self.diffeq_solver = DiffeqSolver(self.ode_func, device=device)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, data, time_points):
        mean, std = self.encoder(data)
        z0 = sample_standard_gaussian(mean, std)
        latent_trajectory = self.diffeq_solver(z0, time_points)
        return self.decoder(latent_trajectory)

def latent_ode_experiment(processed_data, observed_points_list, latent_dim=16, epochs=20, lr=0.01):
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for observed_points in observed_points_list:
        for data in processed_data:
            timestamps = data['timestamps']
            coordinates = data['coordinates']

            observed_indices = np.linspace(0, len(timestamps) - 1, observed_points, dtype=int)
            observed_coordinates = coordinates[observed_indices]
            mean_coords = np.mean(coordinates, axis=0)
            std_coords = np.std(coordinates, axis=0)

            normalized_observed = (observed_coordinates - mean_coords) / std_coords
            normalized_full = (coordinates - mean_coords) / std_coords

            observed_timestamps = torch.tensor(timestamps[observed_indices], dtype=torch.float32).to(device)
            normalized_observed_tensor = torch.tensor(normalized_observed, dtype=torch.float32).unsqueeze(0).to(device)

            model = LatentODESimple(input_dim=2, latent_dim=latent_dim, device=device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                predicted = model(normalized_observed_tensor, observed_timestamps)
                loss = criterion(predicted.squeeze(0), torch.tensor(normalized_full, dtype=torch.float32).to(device))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                predicted_trajectory = model(normalized_observed_tensor, observed_timestamps)
                predicted_trajectory = predicted_trajectory.squeeze(0).cpu().numpy() * std_coords + mean_coords

            results.append({
                'observed_points': observed_points,
                'mmsi': data['mmsi'],
                'full_coordinates': coordinates,
                'predicted_trajectory': predicted_trajectory
            })

    return results
