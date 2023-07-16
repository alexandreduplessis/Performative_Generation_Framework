import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pdb
import datasets
import normflows as nf
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

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers).cuda()

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32).cuda()
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32).cuda() ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # Base Dist for Reward Interpolation
        #self.base_dist = nf.distributions.base.DiagGaussian(2).cuda()
        # Hard code the bounds of the energy
        self.base_dist = nf.distributions.base.Uniform(2, low=-10.0, high=10.0).cuda()
        self.linear_interp = torch.linspace(
                0.1, 1.0, num_timesteps, dtype=torch.float32).cuda()

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def reward_interpolation(self, target_energy, samples, timestep):
        #pdb.set_trace()
        curr_timestep = self.linear_interp[timestep]
        base_energy = self.base_dist.log_prob(samples)
        reward_t = (1 - curr_timestep)*base_energy + curr_timestep * target_energy
        return reward_t

    def cond_step(self, model_output, reward_model, timestep, sample, guidance_weight=1.0):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        with torch.enable_grad():
            pred_prev_sample.requires_grad_()
            #unnorm_pred_prev_sample = (pred_prev_sample * r_std) + r_mean
            # reward = reward_model(unnorm_pred_prev_sample)

            reward = reward_model(pred_prev_sample)
            #reward = self.reward_interpolation(reward, pred_prev_sample, timestep) # We sum because we need scalars
            grad = torch.autograd.grad(reward.sum(), [pred_prev_sample])[0]

            pred_prev_sample.detach()

        variance = 0
        noise = torch.randn_like(model_output)
        if t > 0:
            variance = (self.get_variance(t) ** 0.5)

        pred_prev_sample = pred_prev_sample + variance * noise - grad * variance * guidance_weight
        #print("Max %f and Min %f" %(pred_prev_sample.max(), pred_prev_sample.min()))
        #return torch.clamp(pred_prev_sample, min=-1., max=1.)
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


class SimpleDiffusion():
    """
    Simple DDPM style Diffusion
    """
    def __init__(
            self, device='cpu', num_layers = 5, dim=2, hidden_size=128,
            embedding_size=128, time_embedding='sinusoidal',
            input_embedding='sinusoidal', num_timesteps=150, beta_schedule='linear'):

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

        self.diff_model = MLP(
                hidden_size=hidden_size,
                hidden_layers=num_layers,
                emb_size=embedding_size,
                time_emb=time_embedding,
                input_emb=input_embedding).to(self.device)

        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule)

        self.losses = []
        self.name = f'{self.num_layers}-layers Normalizing Flow'
        self.metrics_titles = {'oldmean': 'Mean error', 'oldstd': 'Standard deviation error', 'oldwasserstein': 'Pseudo-Wasserstein distance',\
                                    'evalmean': 'Mean error', 'evalstd': 'Standard deviation error', 'evalwasserstein': 'Pseudo-Wasserstein distance', 'nll': 'Negative Log-Likelihood'}
        self.max_gradient_norm = 1

    def train(self, data, num)epochs=200):
	global_step = 0
	frames = []
	losses = []
        optimizer = torch.optim.AdamW(
                self.diff_model.parameters())

	print("Training model...")
        progress_bar = tqdm(total=num_epochs)
	for epoch in range(num_epochs):
	    self.diff_model.train()
	    progress_bar.set_description(f"Epoch {epoch}")
	    for step, (unnorm_batch, reward, indices) in enumerate(dataset):
		batch = normalize_batch(unnorm_batch, mins,
                        maxs).to(self.device)
		noise = torch.randn(batch.shape).cuda()
		timesteps = torch.randint(
		    0, noise_scheduler.num_timesteps, (batch.shape[0],)
		).long().to(self.device)

		noisy = self.noise_scheduler.add_noise(batch, noise, timesteps)
		noise_pred = self.diff_model(noisy, timesteps)
		loss = F.mse_loss(noise_pred, noise)
		optimizer.zero_grad()

		if not torch.isnan(loss) and not torch.isinf(loss):
		    loss.backward()
		    nn.utils.clip_grad_norm_(self.diff_model.parameters(), 1.0)
		    optimizer.step()
		else:
		    print("nan loss in non-replay step")

		progress_bar.update(1)
		logs = {"loss": loss.detach().item(), "step": global_step}
		losses.append(loss.detach().item())
		progress_bar.set_postfix(**logs)
		global_step += 1
	    progress_bar.close()

        self.losses = torch.tensor(losses)
	return self.losses

    def generate(self, nb_samples, save_path=None):
        self.diff_model.eval()
        noise_scheduler = NoiseScheduler(
            num_timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule)

        sample = torch.randn(nb_samples, 2).to(self.device)
        timesteps = list(range(len(noise_scheduler)))[::-1]
        frames = []
        samples = []
        steps = []
        plot_step = 50
        for i, t in enumerate(timesteps):
            t = torch.from_numpy(np.repeat(t,
                eval_batch_size)).long().to(self.device)
            with torch.no_grad():
                residual = self.diff_model(sample, t)
            sample = noise_scheduler.step(residual, t[0], sample)
        self.diff_model.train()
        return sample

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
        print("Probability FLOW ODE not implemented. Can only generate samples")
        raise NotImplementedError

    def load(self, path):
        self.diff_model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.diff_model.state_dict(), path)

    def reset(self):
        base_dist = StandardNormal(shape=[self.dim])

        transforms = []
        for _ in range(self.num_layers):
            transforms.append(ReversePermutation(features=self.dim))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.dim,
                                                                hidden_features=64))
        transform = CompositeTransform(transforms)

        self.flow = Flow(transform, base_dist)

