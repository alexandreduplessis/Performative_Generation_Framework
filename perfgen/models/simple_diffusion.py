import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from perfgen.models.ddpm import GaussianDiffusion
import matplotlib.pyplot as plt
import numpy as np
import pdb

# Code taken from https://github.com/tanelp/tiny-diffusion/tree/master
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])).cuda() / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).cuda())
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x)

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb).cuda()
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0).cuda()
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0).cuda()
        self.channels = 1
        self.self_condition = False
        self.out_dim = 1

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers).cuda()

    def forward(self, x, t, self_condition=False):
        # Dims should be B x 2
        x = x.squeeze()
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        x = x.view(-1, 1, 1 , 2)
        return x

class SimpleDiffusion():
    """
    Simple DDPM style Diffusion
    """
    def __init__(
            self, device='cpu', num_layers = 5, dim=2, hidden_size=128,
            embedding_size=128, time_embedding='sinusoidal',
            input_embedding='sinusoidal', num_timesteps=250, beta_schedule='linear'):

        self.num_layers = num_layers
        self.dim = dim
        self.device = device
        self.hidden_size = hidden_size
        self.hidden_layers = num_layers
        self.embedding_size = embedding_size
        self.time_embedding = time_embedding
        self.input_embedding = input_embedding
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule

        self.denoising_model = MLP(
                hidden_size=hidden_size,
                hidden_layers=num_layers,
                emb_size=embedding_size,
                time_emb=time_embedding,
                input_emb=input_embedding).to(self.device)

        self.diffusion = GaussianDiffusion(
                self.denoising_model,
                image_size = self.dim,
                auto_normalize = False,
                timesteps = 1000    # number of steps
            ).to(self.device)

        self.losses = []
        self.name = f'{self.num_layers}-layers Normalizing Flow'
        self.metrics_titles = {'oldmean': 'Mean error', 'oldstd': 'Standard deviation error', 'oldwasserstein': 'Pseudo-Wasserstein distance',\
                                    'evalmean': 'Mean error', 'evalstd': 'Standard deviation error', 'evalwasserstein': 'Pseudo-Wasserstein distance', 'nll': 'Negative Log-Likelihood'}
        self.max_gradient_norm = 1

    def train(self, data, num_epochs=200):
        global_step = 0
        frames = []
        losses = []
        normalized_data, self.mins, self.maxs = self.normalize_dataset(data)
        dataloader = torch.utils.data.DataLoader(
            normalized_data, batch_size=1024, shuffle=True)

        optimizer = torch.optim.AdamW(self.diffusion.model.parameters())
        print("Training model...")
        for epoch in range(num_epochs):
            self.diffusion.model.train()
            if epoch % 20 == 0:
                print("Epoch %d " % (epoch))
            for i, x in enumerate(dataloader):
                optimizer.zero_grad()
                x = x.to(self.device)
                batch = x.to(self.device)
                batch = batch.view(-1, 1 ,1 , 2)
                loss = self.diffusion(batch)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.diffusion.model.parameters(), 1.0)
                    optimizer.step()
                else:
                    print("nan loss in non-replay step")
                logs = {"loss": loss.detach().item(), "step": global_step}
                losses.append(loss.detach().item())
                global_step += 1

            self.losses = torch.tensor(losses)
        return self.losses

    def generate(self, n_samples, save_path=None):
        if n_samples == 0:
            return torch.tensor([])
        self.diffusion.model.eval()
        sample = self.diffusion.sample(n_samples)
        self.diffusion.model.train()
        result = self.unnormalize_dataset(sample.detach(), self.mins, self.maxs)
        return result.reshape(-1, 2)

    def eval(self, data, **kwargs):
        with torch.no_grad():
            metrics = {}
            # compute the std and mean of the data, taking weights into account
            data_gen = self.generate(len(data))
            model_std = torch.std(data_gen, dim=0).cpu()
            metrics['std'] = torch.norm(model_std)
            return metrics

    def log_prob(self, data):
        print("Probability FLOW ODE not implemented. Can only generate samples")
        raise NotImplementedError

    def load(self, path):
        self.diffusion.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.diffusion.state_dict(), path)

    def cold_start(self):
        self.denoising_model = MLP(
            hidden_size=self.hidden_size,
            hidden_layers=self.num_layers,
            emb_size=self.embedding_size,
            time_emb=self.time_embedding,
            input_emb=self.input_embedding).to(self.device)

        self.diffusion = GaussianDiffusion(
                self.denoising_model,
                image_size = self.dim,
                auto_normalize = False,
                timesteps = 1000    # number of steps
            ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.diffusion.model.parameters())

    def normalize_dataset(self, dataset):
        # Normalize data range to [-1, 1] (Assumes min and max data values
        mins = dataset.min(dim=0)[0]
        maxs = dataset.max(dim=0)[0]
        normalized_dataset = (dataset - mins) / (maxs - mins +1e-5)
        normalized_dataset = normalized_dataset * 2 - 1
        print("Max Value %f | Min Value %f" % (normalized_dataset.max(), normalized_dataset.min()))
        return normalized_dataset, mins.to(self.device), maxs.to(self.device)

    def unnormalize_dataset(self, normalized_dataset, mins, maxs):
        normalized_dataset = (normalized_dataset + 1) / 2
        dataset = normalized_dataset * (maxs - mins) + mins
        return dataset
