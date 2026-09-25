import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange

from CustomModules.util import farthest_point_sample
from CustomModules.util import MLP

from sklearn.cluster import KMeans
import numpy as np







class CAE(nn.Module):
    def __init__(self, x_dim, w_dim, z_dim, outer_structure, inner_structure, predictor_structure, chart_amount, activation_outer = nn.ReLU(), activation_inner = nn.ReLU()):
        """
        x_dim = data dimension

        w_dim = middle layer dimension
        
        z_dim = chart space dimension
        """
        super().__init__()
        


        self.chart_amount = chart_amount
        self.x_dim = x_dim # data dimension
        self.w_dim = w_dim # middle layer dimension
        self.z_dim = z_dim # chart space dimension
        self.activation_outer = activation_outer
        self.activation_inner = activation_inner
        self.encoder = MLP([x_dim] + outer_structure + [w_dim], activation=activation_outer)
        self.decoder = MLP([w_dim] + outer_structure + [x_dim], activation=activation_outer)
        self.chart_predictor = MLP([x_dim] + predictor_structure + [chart_amount], activation=activation_outer)

        for chart_index in range(chart_amount):
            self.add_module(f"encoder_{chart_index}", MLP([w_dim] + inner_structure + [z_dim], activation=activation_inner))
            self.add_module(f"decoder_{chart_index}", MLP([z_dim] + inner_structure + [w_dim], activation=activation_inner))



        

    def encode(self, x):
        z_middle = self.encoder(x)
        return z_middle

    def decode(self, z):
        x = self.decoder(z)
        return x

    def predict(self, x):
        logits = self.chart_predictor(x)
        probabililties = F.softmax(logits, dim=1)
        return probabililties

    def chart_encode(self, z, chart_index):
        z_chart = self.get_submodule(f"encoder_{chart_index}")(z)
        return z_chart

    def chart_multi_encode(self, z, chart_indices : torch.Tensor): #for when we want to encode a batch of points with different charts
            z_chart = torch.stack([self.get_submodule(f"encoder_{i}")(z[j]) for j, i in enumerate(chart_indices)])
            return z_chart
    
    
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


    
def pre_train_cae(num_epochs: int, x_train, network : CAE, device: torch.device, lr=1e-3):
    #x= torch.tensor(farthest_point_sample(x_train, k=network.chart_amount)[0], dtype=torch.float32) # (chart_amount, m)
    kmeans = KMeans(n_clusters=network.chart_amount, random_state=np.random.randint(0, 1000), n_init="auto").fit(x_train)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for epoch in trange(num_epochs):
        optimizer.zero_grad()
        loss = 0
        for i in range(network.chart_amount):
            x_sliced = x_train[kmeans.labels_ == i]
            z_middle = network.encode(x_sliced)
            z_latent = network.chart_encode(z_middle, i)
            z_middle_recon = network.chart_decode(z_latent, i) # (1, l)
            x_recon = network.decode(z_middle_recon) # 1, m)
            loss += F.mse_loss(x_recon, x_sliced) - torch.log(network.predict(x_sliced  + 1e-8)[:, i]).mean()

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

            target_probs = F.softmax(-errors.detach(), dim=1)
            
            predicted_probs = network.predict(batch) # (bs, chart_amount)
            log_probs = torch.log(predicted_probs + 1e-8) # (bs, chart_amount)
            
            
            
            loss = torch.min(errors, dim=1).values - torch.sum(target_probs * log_probs, dim=1)  + outer_loss 
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            diag.append((torch.min(errors, dim=1).values, - torch.sum(target_probs * log_probs, dim=1), outer_loss, regularization_loss))
    return diag




    