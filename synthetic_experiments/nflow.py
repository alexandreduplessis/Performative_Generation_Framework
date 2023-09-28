import os
import torch
from torch import optim
from tqdm import tqdm

from utils import plot_data, plot_density
from perf_gen import PerfGenExperiment

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


class MaskedAffineAutoregressiveFlowExperiment(PerfGenExperiment):
    model_name = "MaskedAffineAutoregressiveFlow"
    n_train_iter = 5000
    n_gen_samples = 1000

    def initialize_model(self):
        if os.path.exists(self.get_model_path()) and not self.retrain_initial:
            self.model, self.optim = torch.load(self.get_model_path())
            plot_density(self.train_data, self.model)
            # plot_data(self.train_data, self.generate(500))
            return

        base_dist = StandardNormal(shape=[2])
        transforms = []
        for _ in range(5):
            transforms.append(ReversePermutation(features=2))
            transforms.append(
                MaskedAffineAutoregressiveTransform(features=2, hidden_features=4)
            )
        transform = CompositeTransform(transforms)

        self.model = Flow(transform, base_dist)
        self.optim = optim.Adam(self.model.parameters())

        self.train(self.train_data, 25000)
        torch.save((self.model, self.optim), self.get_model_path())

        print(f"Finished training. Samples look like:")
        # plot_data(self.train_data, self.generate(500))
        plot_density(self.train_data, self.model)

    def generate(self, n_samples):
        return self.model.sample(n_samples).detach()

    def train(self, data, n_train_iter):
        for i in tqdm(range(n_train_iter)):
            self.optim.zero_grad()
            loss = -self.model.log_prob(inputs=data).mean()
            loss.backward()
            self.optim.step()
