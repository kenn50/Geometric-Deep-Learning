import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from sklearn.neighbors import NearestNeighbors

from GDLModules.util import *


from sklearn.cluster import KMeans
import numpy as np







class CAE(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_structure, predictor_structure, chart_amount, activation = nn.ReLU(), 
                autoencoder = None):
        """
        x_dim = data dimension

        z_dim = chart space dimension
        """
        super().__init__()

        

        self.autoencoder = autoencoder
        self.chart_amount = chart_amount
        self.x_dim = x_dim # data dimension
        self.z_dim = z_dim # chart space dimension
        self.activation_inner = activation

        self.chart_predictor = MLP(x_dim, chart_amount, predictor_structure, activation=activation)

        input_dim = x_dim if not autoencoder else autoencoder.z_dim


        for chart_index in range(chart_amount):
            self.add_module(f"encoder_{chart_index}", MLP(input_dim, z_dim, hidden_structure, activation=activation))
            self.add_module(f"decoder_{chart_index}", MLP(z_dim, input_dim, hidden_structure, activation=activation))

    def true_predict(self, x):
        network = self
        data = x
        device = x.device
        errors = torch.zeros((len(data),network.chart_amount), device=device) # (chart_amount, d)
        for i in range(network.chart_amount):
            z = network.encode(data, i)
            decoded= network.decode(z, i) # (bs, D)
            errors[:, i] = norm_squared(decoded, data, dim=1)

        probs = F.softmax(-errors.detach(), dim=1)
        return probs


 
    def predict(self, x):
        logits = self.chart_predictor(x)
        probabililties = F.softmax(logits, dim=1)
        return probabililties
    
    def true_predict(self, x):
        errors = torch.zeros((len(x),self.chart_amount), device=x.device)
        
        for i in range(self.chart_amount):
            z = self.encode(x, i)
            decoded = self.decode(z, i) # (bs, D)
            errors[:, i] = norm_squared(decoded, x, dim=1)
        
        probabililties = F.softmax(-errors.detach(), dim=1)
        return probabililties
        
         
        

    def encode(self, x, chart_index):
        #if autoencoder is available, we use it first:
        network_input = x if not self.autoencoder else self.autoencoder.encode(x)
        chart_index = torch.tensor(chart_index, device=x.device)

        if chart_index.ndim == 0:
            z = self.get_submodule(f"encoder_{chart_index.item()}")(network_input)
        else:
            z = torch.stack([self.get_submodule(f"encoder_{i}")(network_input[j]) for j, i in enumerate(chart_index)])

        return z
    
    def decode(self, z, chart_indexes):
        chart_indexes = torch.tensor(chart_indexes, device=z.device)

        if chart_indexes.ndim == 0:
            recon = self.get_submodule(f"decoder_{chart_indexes.item()}")(z)
        else:
            recon = torch.stack([self.get_submodule(f"decoder_{i}")(z[j]) for j, i in enumerate(chart_indexes)])

        x = self.autoencoder.decode(z) if self.autoencoder else recon
        return x




    
def pre_train_cae(num_epochs: int, x_train, network : CAE, device: torch.device, knn = 100, lr=1e-3):
    furthest_points = farthest_point_sample(x_train, k=network.chart_amount)[0] # (chart_amount, m)
    nbs = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(x_train)
    
    #kmeans = KMeans(n_clusters=network.chart_amount, random_state=np.random.randint(0, 1000), n_init="auto").fit(x_train)
    
    x_train = x_train.to(device)
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for epoch in trange(num_epochs):
        optimizer.zero_grad()
        loss = 0
        for i in range(network.chart_amount):
            distances, indices = nbs.kneighbors(furthest_points[i].reshape(1, -1))
            x_sliced = x_train[indices.flatten()]
            #x_sliced = x_train[kmeans.labels_ == i]
            
            z = network.encode(x_sliced, i)
            x_recon = network.decode(z, i) # 1, m)
            loss += F.mse_loss(x_recon, x_sliced) - torch.log(network.predict(x_sliced  + 1e-8)[:, i]).mean()


        loss.backward()
        optimizer.step()

def train_cae(num_epochs: int, x_train_loader: DataLoader, network: CAE, device: torch.device, lr=1e-4, mmd_weight=1.0):
    network.to(device)
    diag = []
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    pbar = trange(num_epochs)

    losses = torch.zeros((10,))
    losses_index = 0
    
    for epoch in pbar:
        for batch in x_train_loader:
            
            batch = batch.to(device)
            optimizer.zero_grad()
            autoencoder_loss = 0
            # if network.autoencoder:
            #     z_middle = network.autoencoder.encode(batch) # (bs, d)
            #     decoded = network.autoencoder.decode(z_middle) # (bs, D)
            #     autoencoder_loss = F.mse_loss(decoded, batch)
            errors = torch.zeros((len(batch),network.chart_amount), device=device) # (chart_amount, d)
            zs = torch.zeros((len(batch), network.chart_amount, network.z_dim), device=device) # (chart_amount, d, z_dim)
            regularization_loss = 0
            mmd_loss = 0
            for i in range(network.chart_amount):
                z = network.encode(batch, i) # (bs, z_dim)
                decoded = network.decode(z, i) # (bs, D)
                errors[:, i] = norm_squared(decoded, batch, dim=1)
                
                zs[:, i, :] = z

                    

           

            target_probs = F.softmax(-errors.detach(), dim=1)
            
            
            
            predicted_probs = network.predict(batch) # (bs, chart_amount)
            log_probs = torch.log(predicted_probs + 1e-8) # (bs, chart_amount)

            k1 = torch.kthvalue(errors, 1, dim=1)[0] # (bs,)
            k2 = torch.kthvalue(errors, 2, dim=1)[0]
            index = torch.concat([k1.unsqueeze(1), k2.unsqueeze(1)], dim=1) # (bs, 2)
            index = index.unsqueeze(-1).expand(-1, -1, zs.size(2)) # (bs, 2, z_dim)
            
            index = index.long()
            
            mmd_input = zs.gather(dim=1, index=index).reshape(-1, zs.size(2)) # (bs*2, z_dim)
            mmd_loss = MMD_score(mmd_input, torch.randn_like(mmd_input)) # (bs*2, z_dim)
            
            

            
            l1 = k1[0] + k2[0] # (bs,)
            loss =  l1 - torch.sum(target_probs * log_probs, dim=1)  + autoencoder_loss + mmd_weight * mmd_loss  # (bs,)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            
            losses[losses_index] = loss.item()
            losses_index = (losses_index + 1) % len(losses)

            pbar.set_postfix(loss=f"{losses.mean().item():.4f}")


            diag.append((torch.min(errors, dim=1).values, - torch.sum(target_probs * log_probs, dim=1), autoencoder_loss, regularization_loss))
    return diag




    