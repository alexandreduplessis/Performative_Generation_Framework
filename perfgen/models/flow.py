import numpy as np
from perfgen.utils import wasserstein_distance
import torch
from torch import nn
from torch import optim
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

class Normalizing_Flow():
    """
    Normalizing Flow
    """
    def __init__(
            self, num_layers = 5, dim=2):
        self.num_layers = num_layers
        self.dim = dim
        base_dist = StandardNormal(shape=[self.dim])

        transforms = []
        for _ in range(self.num_layers):
            transforms.append(ReversePermutation(features=self.dim))
            transforms.append(MaskedAffineAutoregressiveTransform(
                features=self.dim,
                hidden_features=64))
        transform = CompositeTransform(transforms)

        self.flow = Flow(transform, base_dist)
        self.losses = []
        self.name = f'{self.num_layers}-layers Normalizing Flow'
        self.metrics_titles = {'oldmean': 'Mean error', 'oldstd': 'Standard deviation error', 'oldwasserstein': 'Pseudo-Wasserstein distance',\
                                    'evalmean': 'Mean error', 'evalstd': 'Standard deviation error', 'evalwasserstein': 'Pseudo-Wasserstein distance', 'nll': 'Negative Log-Likelihood'}
        self.max_gradient_norm = 1

    def train(self, data, epochs=10_000):
        dataloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
        optimizer = optim.Adam(self.flow.parameters())
        self.losses = []
        for epoch in range(epochs):
            loss_sum = 0
            for x in dataloader:
                optimizer.zero_grad()
                loss = -self.flow.log_prob(inputs=x).mean()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.flow.parameters(),
                    self.max_gradient_norm)
                loss_sum += loss.item()
                if torch.isfinite(grad_norm):
                    optimizer.step()
                self.losses.append(loss_sum)
        self.losses = torch.tensor(self.losses)
        return self.losses

    def generate(self, nb_samples, save_path=None):
        if nb_samples == 0:
            return torch.tensor([])
        samples = self.flow.sample(nb_samples).detach()
        if save_path is not None:
            # torch save
            torch.save(samples, save_path)
        return samples

    def eval(self, data, **kwargs):
        with torch.no_grad():
            metrics = {}
            # compute the std and mean of the data, taking weights into account
            data_gen = self.generate(len(data))
            model_std = torch.std(data_gen, dim=0)
            metrics['std'] = torch.norm(model_std)
            metrics['wasserstein'] = wasserstein_distance(data, data_gen)
            metrics['nll'] = - self.log_prob(data).mean(axis=-1).detach().numpy()
            return metrics

    def log_prob(self, data):
        return self.flow.log_prob(data)

    def load(self, path):
        self.flow.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.flow.state_dict(), path)

    def reset(self):
        base_dist = StandardNormal(shape=[self.dim])

        transforms = []
        for _ in range(self.num_layers):
            transforms.append(ReversePermutation(features=self.dim))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.dim,
                                                                hidden_features=64))
        transform = CompositeTransform(transforms)

        self.flow = Flow(transform, base_dist)
