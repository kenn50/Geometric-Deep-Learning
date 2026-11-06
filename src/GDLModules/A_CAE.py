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
    def __init__(self, x_dim, z_dim, hidden_structure, chart_amount, activation = nn.ReLU()):
        """
        x_dim = data dimension

        z_dim = chart space dimension
        """
        super().__init__()

        self.chart_amount = chart_amount
        self.x_dim = x_dim # data dimension
        self.z_dim = z_dim # chart space dimension
        self.activation_inner = activation

        input_dim = x_dim
        
        self.encoders = nn.ModuleList([MLP(input_dim, z_dim, hidden_structure, activation=activation) for _ in range(chart_amount)])
        self.decoders = nn.ModuleList([MLP(z_dim, input_dim, hidden_structure, activation=activation) for _ in range(chart_amount)])
    

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



    
    def true_predict(self, x):
        errors = torch.zeros((len(x),self.chart_amount), device=x.device)
        
        for i in range(self.chart_amount):
            z = self.encode(x, i)
            decoded = self.decode(z, i) # (bs, D)
            errors[:, i] = norm_squared(decoded, x, dim=1)
        
        probabililties = F.softmax(-errors.detach(), dim=1)
        return probabililties
        
         
        

    def encode(self, x, chart_index):
        network_input = x
        chart_index = torch.tensor(chart_index, device=x.device)

        if chart_index.ndim == 0:
            z = self.encoders[chart_index.item()](network_input)
        else:
            z = torch.stack([self.encoders[i](network_input[j]) for j, i in enumerate(chart_index)])

        return z
    
    def decode(self, z, chart_indexes):
        chart_indexes = torch.tensor(chart_indexes, device=z.device)

        if chart_indexes.ndim == 0:
            recon = self.decoders[chart_indexes.item()](z)
        else:
            recon = torch.stack([self.decoders[i](z[j]) for j, i in enumerate(chart_indexes)])

        x = recon
        return x




def get_lipschitz_loss(module_list, device):
    loss = 0
    num_charts = len(module_list)
    
    #Count layers
    num_layers = 0
    for module in module_list:
        for layer in module.network:
            if isinstance(layer, nn.Linear):
                num_layers+=1
    
    m = torch.zeros((num_charts, num_layers), device=device)
    
    for i, module in enumerate(module_list):
            j = 0
            for layer in module.network:
                if isinstance(layer, nn.Linear):
                    m[i, j] = torch.log(torch.linalg.matrix_norm(layer.weight, ord=2))
                    
                    j+=1
          

    products = torch.abs(1-torch.exp(torch.sum(m, dim=1)))
                
    loss = torch.sum(products)/num_charts + torch.max(products)
                
                
    return loss


    
def pre_train_cae(num_epochs: int, x_train, network : CAE, device: torch.device, knn = 100, lr=1e-3):
    furthest_points = farthest_point_sample(x_train.cpu().detach().numpy(), k=network.chart_amount)[0] # (chart_amount, D)
    nbs = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(x_train.cpu().detach().numpy())

    
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
            
            # Generate 1000 points normally disttribured around furthest_points[i]. The normal distrubtuion should be a z_dim dimensional manifold with a random rotation
            center = torch.as_tensor(furthest_points[i], device=device, dtype=x_train.dtype)
            x_normal = center + 0.5 * torch.randn((len(x_sliced), network.x_dim), device=device, dtype=x_train.dtype)
            
            z = network.encode(x_normal, i)
            x_recon = network.decode(z, i) # 1, m)
            loss += F.mse_loss(x_recon, x_normal)


        loss.backward()
        optimizer.step()

def train_cae(num_epochs: int, x_train_loader: DataLoader, network: CAE, device: torch.device, lr=1e-4, best_k=1, mmd_weight=0.0, lipschitz_weight=0.0, stop_event=None, threshold=5e-2):
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=100)
    
    pbar = trange(num_epochs)

    losses = torch.zeros((10,))
    losses_index = 0
    
    
    
    for epoch in pbar:
        if stop_event is not None and stop_event.is_set():
            print("Training stopped.")
            break
        for batch in x_train_loader:
            
            batch = batch.to(device)
            optimizer.zero_grad()
            errors = torch.zeros((len(batch),network.chart_amount), device=device) # (bs, chart_amount)
            zs = torch.zeros((len(batch), network.chart_amount, network.z_dim), device=device) # (bs, chart_amount, z_dim)
            mmd_loss = 0
            for i in range(network.chart_amount):
                z = network.encode(batch, i) # (bs, z_dim)
                decoded = network.decode(z, i) # (bs, D)
                errors[:, i] = norm_squared(decoded, batch, dim=1)
                
                zs[:, i, :] = z



            kevins_unsummed_loss = nn.functional.relu(threshold - errors) # (bs, chart_amount)

            # if two charts or more are both > 0 on a point, remove the reward entirely for that point
            kevins_usummed_punish_loss = kevins_unsummed_loss.clone()
            kevins_usummed_punish_loss[kevins_usummed_punish_loss.gt(0).sum(dim=1) > 1] = 0
            
            kevins_loss = kevins_unsummed_loss.sum()
            


            chart_losses = []
            for i in range(network.chart_amount):
                #for every point > 0, get the index
                column = kevins_unsummed_loss[:, i]
                gt0 = column.gt(0)
                if gt0.any():
                    rewarded_points = zs[gt0, i] # (bs, z_dim)
                    chart_losses.append(MMD_score(rewarded_points, torch.randn_like(rewarded_points)))

            mmd_loss = torch.stack(chart_losses).mean()
            


            
            l1 = -kevins_loss
            
            
            loss =  l1  + mmd_weight * mmd_loss + lipschitz_weight * get_lipschitz_loss(network.encoders, device)   # (bs,)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            losses[losses_index] = loss.item()
            losses_index = (losses_index + 1) % len(losses)

            pbar.set_postfix(rec_loss=f"{100*l1.mean().item():.4f}%",mmd_loss=f"{100*mmd_loss.item():.4f}%", loss=f"{100*losses.mean().item():.4f}%", lr=f"{optimizer.param_groups[0]['lr']:.6f}")





    
