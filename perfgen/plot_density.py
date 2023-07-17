import numpy as np
import torch
import wandb
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt

def plt_density(
        model, npts=100, memory=100, title="$q(x)$", device="cpu", LOW = -4, HIGH = 4, plt_name = "Default"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    z = torch.tensor(z, requires_grad=False).type(torch.float32).to(device)

    with torch.no_grad():
        log_prob = model.log_prob(z)
        prob = torch.exp(log_prob).reshape(npts, npts)
    prob = prob.cpu().numpy()
    # ax.imshow(prob)
    plt.imshow(prob, cmap=cm.magma)
    plt.axis('off')
    # set size
    plt.gcf().set_size_inches(5, 5)
    # plt.set_title(title)
    wandb.log({plt_name: plt})

def plot_samples(
        dataset, gen_samples, npts=100, memory=100, title="$q(x)$", device="cpu", LOW = -4, HIGH = 4, plt_name = "Default"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
    axs[0].scatter(dataset[:, 0], dataset[:, 1], alpha=0.5, s=15)
    axs[1].scatter(gen_samples[:, 0], gen_samples[:, 1], alpha=0.5, s=15)
    axs[0].set_xlim([-4, 4])
    axs[0].set_ylim([-4, 4])
    plt.gcf().set_size_inches(5, 5)
    wandb.log({plt_name: wandb.Image(plt)})
