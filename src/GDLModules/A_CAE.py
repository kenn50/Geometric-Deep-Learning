import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from sklearn.neighbors import NearestNeighbors

from GDLModules.util import sample_uniform_ball_like, MMD_score, ChamferLoss
from GDLModules.util import *

from geomloss import SamplesLoss

from sklearn.cluster import KMeans
import numpy as np

from torch.func import jacfwd
from torch_pca import PCA as torchPCA
from torch.nn.utils.parametrizations import orthogonal






class Chart(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_structure, activation=nn.ReLU(), activation_last_layer=None):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        
        self.encoder = MLP(x_dim, z_dim, hidden_structure, activation=activation, activation_last_layer=activation_last_layer)
        self.decoder = MLP(z_dim, x_dim, hidden_structure, activation=activation, activation_last_layer=activation_last_layer, last_zero=True)
        
        self.affine_center = nn.Parameter(torch.randn(x_dim))
        self.affine_linear = orthogonal(
            nn.Linear(x_dim, x_dim, bias=False)
        )
    
    
    
    def encode(self, x):
        A = self.affine_linear.weight
        x_d = (x - self.affine_center) @ A
        return self.encoder(x_d)
    
    
    def decode(self, z):
        z_padded = F.pad(z, (0, self.x_dim - self.z_dim))
        x_d = self.decoder(z) + z_padded

        A = self.affine_linear.weight
        return x_d @ A.T + self.affine_center
    
    

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
        self.hidden_structure = hidden_structure

        
        self.charts = nn.ModuleList([Chart(x_dim, z_dim, hidden_structure, activation=activation) for _ in range(chart_amount)])

    
    
    def new_chart(self, device):
        new_chart = Chart(self.x_dim, self.z_dim, self.hidden_structure, activation=self.activation_inner).to(device)
        self.charts.append(new_chart)
        self.chart_amount += 1
        return new_chart


    def true_predict(self, x):
        errors = torch.zeros((len(x),self.chart_amount), device=x.device)
        
        for i in range(self.chart_amount):
            z = self.encode(x, i)
            decoded = self.decode(z, i) # (bs, D)
            errors[:, i] = norm_squared(decoded, x, dim=1)
        
        probabililties = F.softmax(-errors.detach(), dim=1)
        return probabililties
        
         
        

    def encode(self, x, chart_index):
        chart_index = torch.tensor(chart_index, device=x.device)

        if chart_index.ndim == 0:
            z = self.charts[chart_index.item()].encode(x)
        else:
            z = torch.stack([self.charts[i].encode(x[j]) for j, i in enumerate(chart_index)])

        return z
    
    def decode(self, z, chart_indexes):
        chart_indexes = torch.tensor(chart_indexes, device=z.device)
        
        if chart_indexes.ndim == 0:
            recon = self.charts[chart_indexes.item()].decode(z)
        else:
            recon = torch.stack([self.charts[i].decode(z[j]) for j, i in enumerate(chart_indexes)])

        x = recon
        return x




def get_lipschitz_loss(module_list, device):
    loss = 0
    num_charts = len(module_list)
    
    #Count layers
    num_layers = 0
    for chart_module in module_list:
        for layer in chart_module.network:
            if isinstance(layer, nn.Linear):
                num_layers+=1
    
    m = torch.zeros((num_charts, num_layers), device=device)
    
    for chart_index, chart_module in enumerate(module_list):
            layer_index = 0
            for layer in chart_module.network:
                if isinstance(layer, nn.Linear):
                    m[chart_index, layer_index] = torch.log(torch.linalg.matrix_norm(layer.weight, ord=2))
                    
                    layer_index+=1
          

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
            x_normal = center + 0.1 * torch.randn((len(x_sliced), network.x_dim), device=device, dtype=x_train.dtype)
            
            z = network.encode(x_normal, i)
            x_recon = network.decode(z, i) # 1, m)
            loss += F.mse_loss(x_recon, x_normal)


        loss.backward()
        optimizer.step()






def get_jacobian(f, points, return_function_output = False):
    in_f = f
    if return_function_output:
        def double_f(x):
            out = f(x)
            return out, out
        in_f = double_f
    
    return torch.vmap(jacfwd(in_f, has_aux=return_function_output))(points)
    





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
            
            kevins_loss = kevins_unsummed_loss.sum(dim=1)
            


            chart_losses = []
            for i in range(network.chart_amount):
                #for every point > 0, get the index
                column = kevins_unsummed_loss[:, i]
                gt0 = column.gt(0)
                if gt0.any():
                    rewarded_points = zs[gt0, i] # (bs, z_dim)
                    target = sample_uniform_ball_like(rewarded_points, radius=1.0)
                    chart_losses.append(MMD_score(rewarded_points, target))

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








def reverse_train_cae(num_epochs: int, x_train, network: CAE, device: torch.device, lr=1e-4, lipschitz_weight=0.0, norm_threshold = 2.0, stop_event=None, batch_size=512, spawn_interval=100, criterion_type="chamfer"):
    
    
    
    if criterion_type == "sinkhorn":
        criterion = SamplesLoss(
            loss="sinkhorn", p=2, blur=0.05, scaling=0.8, backend="tensorized", debias=True
        )
    elif criterion_type == "chamfer":
        criterion = ChamferLoss()
    else:
        assert False, "Invalid criterion type"
    
    
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    pbar = trange(num_epochs)
    x_train = x_train.to(device)
    x_train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
    for epoch in pbar:
        if stop_event is not None and stop_event.is_set():
            print("Training stopped.")
            break
         
        for batch_idx, batch in enumerate(x_train_loader):
            
            
            # sample 1000 points
            
            optimizer.zero_grad()
            z_sample = torch.rand(512, network.z_dim).to(device)
            encoder_loss = 0
            svd_reg = 0
            decoded_lst = []
            for i in range(network.chart_amount):
                J, x_decoded = get_jacobian(network.charts[i].decode, z_sample, return_function_output=True)
                decoded_lst.append(x_decoded)
                
                #penalize a large ratio
                svdvals= torch.linalg.svdvals(J)
                
                svd_reg += (svdvals - 1).square().sum()
                
                #Reconstruction loss for encoder

                z_encoded = network.charts[i].encode(x_decoded)
                encoder_loss += F.mse_loss(z_encoded, z_sample)
                
    
            
            decoded = torch.vstack(decoded_lst)
            loss = criterion(decoded, batch) + encoder_loss + lipschitz_weight*svd_reg
            pbar.set_postfix(loss=loss.item(), chart_amount=network.chart_amount)
            loss.backward()
            optimizer.step()
            
            
            # At the start of each epoch, spawn a new chart
            
            if batch_idx==0 and epoch % spawn_interval == 0 and epoch > 0:
                print("spawning")
                
                with torch.no_grad():
                    
                    chart_scores = torch.zeros(len(batch), network.chart_amount, device=device)
                    for i in range(network.chart_amount):

                        J, z_encoded_batch = get_jacobian(network.charts[i].encode, batch, return_function_output=True)
                        J_norm = torch.linalg.matrix_norm(J, ord=2) # (bs,)
                        x_decoded_batch = network.decode(z_encoded_batch, i)
                        mse = F.mse_loss(
                            batch,
                            x_decoded_batch,
                            reduction="none"
                        ).mean(dim=1)
                        
                        chart_scores[:, i] = J_norm + mse

                    
                    #find the points, which is handled worst by model
                    chart_scores_best = chart_scores.min(dim=1).values
                    worst_point_idx = torch.argmax(chart_scores_best)
                    worst_point = batch[worst_point_idx].clone().detach()
                    
                    #spawn new chart at this point
                    new_chart = network.new_chart(device)
                    
                    
                    new_chart.affine_center.copy_(worst_point)
                    
                    dist = torch.norm(batch - worst_point, dim=1, p=None)
                    knn = dist.topk(50, largest=False)
                    pca = torchPCA(n_components=network.x_dim)
                    pca.fit(batch[knn.indices])
                    A = pca.components_.T # Note that the pca returns pcs as rows, so we need to transpose it to get the correct shape
                    
                    new_chart.affine_linear.weight = A.to(device, dtype=new_chart.affine_linear.weight.dtype)
                    
                    
                    optimizer.add_param_group({
                        "params": new_chart.parameters()
                    })
                                        
                    

            
    return

def detect_problem_area():
    pass


