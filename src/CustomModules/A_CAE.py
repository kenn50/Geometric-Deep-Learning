import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange

from CustomModules.util import farthest_point_sample
from CustomModules.util import FFNN







class CAE(nn.Module):
    def __init__(self, m, l, d, outer_structure, inner_structure, predictor_structure, chart_amount, activation_outer = nn.ReLU(), activation_inner = nn.ReLU()):
        super().__init__()
        self.chart_amount = chart_amount
        self.m = m
        self.l = l
        self.d = d
        self.activation_outer = activation_outer
        self.activation_inner = activation_inner
        self.encoder = FFNN([m] + outer_structure + [l], activation=activation_outer)
        self.decoder = FFNN([l] + outer_structure + [m], activation=activation_outer)
        self.chart_predictor = FFNN([m] + predictor_structure + [chart_amount], activation=activation_outer)

        for alpha in range(chart_amount):
            self.add_module(f"encoder_{alpha}", FFNN([l] + inner_structure + [d], activation=activation_inner))
            self.add_module(f"decoder_{alpha}", FFNN([d] + inner_structure + [l], activation=activation_inner))


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
            z_alpha = torch.stack([self.get_submodule(f"encoder_{i}")(z[j]) for j, i in enumerate(alpha)])
            return z_alpha
    
    
    def chart_decode(self, z_alpha, alpha):
        z = self.get_submodule(f"decoder_{alpha}")(z_alpha)
        return z

    def chart_multi_decode(self, z_alpha, alpha : torch.Tensor): #for when we want to encode a batch of points with different charts
            z_recon = torch.stack([self.get_submodule(f"decoder_{i}")(z_alpha[j]) for j, i in enumerate(alpha)])
            return z_recon

    def encode_decode(self, x):
        z_middle = self.encode(x)
        probabilities = self.predict(x)
        alpha = torch.argmax(probabilities, dim=1)
        z_latent = self.chart_multi_encode(z_middle, alpha)
        z_recon = self.chart_multi_decode(z_latent, alpha)
        x_recon = self.decode(z_recon)
        return x_recon










    
def pre_train_cae(num_epochs: int, x_train, network : CAE, device: torch.device):
    x= torch.tensor(farthest_point_sample(x_train, k=network.chart_amount)[0], dtype=torch.float32) # (chart_amount, m)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        z_middle = network.encode(x) # (chart_amount, l)
        loss = 0
        for i in range(network.chart_amount):
            z_latent = network.chart_encode(z_middle[i], i).unsqueeze(0) # (d) => unsqueeze to (1, d)
            assert(z_latent.shape[0] == 1)
            z_middle_recon = network.chart_decode(z_latent, i) # (1, l)
            x_recon = network.decode(z_middle_recon) # 1, m)
            loss += F.mse_loss(x_recon, x[i]) + F.mse_loss(z_latent, torch.zeros_like(z_latent)+0.5) - torch.log(network.predict(x)[i, i] + 1e-8)

        loss.backward()
        optimizer.step()

def train_cae(num_epochs: int, x_train_loader: DataLoader, network: CAE, device: torch.device, lr=1e-4):
    network.to(device)
    diag = []
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for epoch in trange(num_epochs):
        for batch in x_train_loader:
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            z_middle = network.encode(batch) # (bs, d)
            decoded = network.decode(z_middle) # (bs, D)
            outer_loss = F.mse_loss(decoded, batch) 
            errors = torch.zeros((len(batch),network.chart_amount), device=device) # (chart_amount, d)
            regularization_loss = 0
            for i in range(network.chart_amount):
                z_latent = network.chart_encode(z_middle, i)
                z_middle_recon = network.chart_decode(z_latent, i)
                decoded = network.decode(z_middle_recon) # (bs, D)
                errors[:, i] = torch.norm((decoded - batch), dim = 1)

                sampled_points_x= torch.rand((100, network.d)).to(device) # (10, d)
                sampled_points_y = torch.roll(sampled_points_x, shifts=1, dims=0)
                sampled_diff = sampled_points_x - sampled_points_y

                decoded_sampled_x = network.chart_decode(sampled_points_x, i)
                decoded_sampled_y = torch.roll(decoded_sampled_x, shifts=1, dims=0)
                decoded_diff = decoded_sampled_x - decoded_sampled_y

                regularization_loss += F.mse_loss(torch.log(1e-8 + torch.sum(sampled_diff * sampled_diff, dim=1)), torch.log(1e-8 + torch.sum(decoded_diff * decoded_diff, dim=1)))


            target_probs = F.softmax(-errors.detach(), dim=1)
            
            predicted_probs = network.predict(batch) # (bs, chart_amount)
            log_probs = torch.log(predicted_probs + 1e-8) # (bs, chart_amount)
            
            
            
            loss = torch.min(errors, dim=1).values - torch.sum(target_probs * log_probs, dim=1)  + outer_loss  #+ regularization_loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            diag.append((torch.min(errors, dim=1).values, - torch.sum(target_probs * log_probs, dim=1), outer_loss, regularization_loss))
    return diag




    