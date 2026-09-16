import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange

from CustomModules.util import farthest_point_sample









class CAE(nn.Module):
    def __init__(self, m, l, d, hidden_dim, hidden_chart_dim, hidden_predictor_dim, chart_amount):
        super().__init__()
        self.chart_amount = chart_amount
        self.encoder = nn.Sequential(nn.Linear(m, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, l))
        self.decoder = nn.Sequential(nn.Linear(l, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, m))
        self.chart_predictor = nn.Sequential(nn.Linear(m, hidden_dim),nn.ReLU(), nn.Linear(hidden_dim, chart_amount))
        

        for alpha in range(chart_amount):
            self.add_module(f"encoder_{alpha}", nn.Sequential(nn.Linear(l, hidden_chart_dim), nn.ReLU(), nn.Linear(hidden_chart_dim, d)))
            self.add_module(f"decoder_{alpha}", nn.Sequential(nn.Linear(d, hidden_chart_dim), nn.ReLU(), nn.Linear(hidden_chart_dim, l)))


    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x

    def predict(self, x):
        logits = self.chart_predictor(x)
        probabililties = F.softmax(logits, dim=1)
        return probabililties

    def chart_encode(self, z, alpha):
        z_alpha = self.get_submodule(f"encoder_{alpha}")(z)
        return z_alpha

    def chart_multi_encode(self, z, alpha : torch.Tensor): #for when we want to encode a batch of points with different charts
            z_alpha = torch.stack([self.get_submodule(f"encoder_{i}")(z[j].unsqueeze(0)) for j, i in enumerate(alpha)])
            return z_alpha
    
    
    def chart_decode(self, z_alpha, alpha):
        z = self.get_submodule(f"decoder_{alpha}")(z_alpha)
        return z

    def chart_multi_decode(self, z_alpha, alpha : torch.Tensor): #for when we want to encode a batch of points with different charts
            z_recon = torch.stack([self.get_submodule(f"decoder_{i}")(z_alpha[j].unsqueeze(0)) for j, i in enumerate(alpha)])
            return z_recon

    def encode_decode(self, x):
        z_middle = self.encode(x)
        probabilities = self.predict(x)
        alpha = torch.argmax(probabilities, dim=1)
        z_latent = self.chart_multi_encode(z_middle, alpha)
        z_recon = self.chart_multi_decode(z_latent, alpha)
        x_recon = self.decode(z_recon)
        return x_recon










    
def pre_train_cae(epochs: int, x_train, network : CAE, device: torch.device):
    x, indices = farthest_point_sample(x_train, K=network.chart_amount)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    for epoch in range(epochs):
        optimizer.zero_grad()
        z_middle = network.encode(x)
        loss = 0
        for i in range(network.chart_amount):
            z_latent = network.chart_encode(z_middle[i].unsqueeze(0), i)
            z_middle_recon = network.chart_decode(z_latent, i)
            x_recon = network.decode(z_middle_recon)
            loss += F.mse_loss(x_recon, x[i]) + F.mse_loss(z_latent, torch.zeros_like(z_latent)+0.5) - torch.log(network.predict(x[i])[i] + 1e-8)

        loss.backward()
        optimizer.step()

def train_cae(epochs: int, x_train_loader: DataLoader, network: CAE, device: torch.device):
    network.to(device)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    for epoch in trange(epochs):
        for batch in x_train_loader:
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            z_middle = network.encode(batch) # (bs, d)
            errors = torch.zeros((len(batch),network.chart_amount), device=device) # (chart_amount, d)

            z_middle_losses = 0
    
            for i in range(network.chart_amount):
                z_latent = network.chart_encode(z_middle, i)
                z_middle_recon = network.chart_decode(z_latent, i)
                decoded = network.decode(z_middle_recon) # (bs, D)
                errors[:, i] = torch.norm((decoded - batch), dim = 1)
                z_middle_losses += F.mse_loss(z_middle, z_middle_recon)


            predicted_probs = network.predict(batch) # (bs, chart_amount)
            log_probs = torch.log(predicted_probs + 1e-8) # (bs, chart_amount)
            
            loss = torch.min(errors, dim=1).values - torch.sum(F.softmax(-errors, dim = 1) * log_probs, dim=1) + z_middle_losses
            loss = loss.mean()
            loss.backward()
            optimizer.step()
    




    