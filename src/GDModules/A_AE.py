import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange

from CustomModules.util import farthest_point_sample

class AE(nn.Module):
    def __init__(self, m, d, hidden_dim):
        super().__init__()
        self.m = m
        self.d = d
        self.encoder = nn.Sequential(nn.Linear(m, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d))
        self.decoder = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, m))


    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x


    def encode_decode(self, x):
        z_latent = self.encode(x)
        x_recon = self.decode(z_latent)
        return x_recon



def train_ae(num_epochs: int, x_train_loader: DataLoader, network: AE, device: torch.device):
    network.to(device)
    diag = []
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    for epoch in trange(num_epochs):
        for batch in x_train_loader:
            
            batch = batch.to(device)
            optimizer.zero_grad()
            z = network.encode(batch) 
            recon = network.decode(z) 
            loss = F.mse_loss(recon, batch) 
            loss.backward()
            optimizer.step()
    return




