import numpy as np
import torch
import wandb
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib import rc


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
    # plt.gcf().set_size_inches(5, 5)
    # plt.set_title(title)
    wandb.log({plt_name: plt})


def plot_kde_density(
        dataset, gen_samples, npts=100, memory=100, title="$q(x)$", device="cpu", LOW = -4, HIGH = 4, plt_name = "Default"):

    _, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
    kernel_1 = stats.gaussian_kde(dataset.T, bw_method=0.1)
    kernel_2 = stats.gaussian_kde(gen_samples.T, bw_method=0.1)

    X, Y = np.mgrid[LOW:HIGH:200j, LOW:HIGH:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z1 = np.reshape(kernel_1(positions).T, X.shape)
    Z2 = np.reshape(kernel_2(positions).T, X.shape)

    axs[0].pcolormesh(X, Y, Z1, cmap='magma')
    axs[1].pcolormesh(X, Y, Z2, cmap='magma')

    axs[0].set_xlim([-4, 4])
    axs[0].set_ylim([-4, 4])

    axs[1].set_xlim([-4, 4])
    axs[1].set_ylim([-4, 4])

    cmap = matplotlib.cm.get_cmap(None)

    axs[0].set_facecolor(cmap(0.))
    axs[0].invert_yaxis()
    axs[1].set_facecolor(cmap(0.))
    axs[1].invert_yaxis()
    axs[0].set_title("Ground Truth")
    axs[1].set_title("KDE Density on Gen. Data")
    # plt.gcf().set_size_inches(5, 5)
    wandb.log({plt_name: wandb.Image(plt)})


def plot_samples(
        dataset, gen_samples, npts=100, memory=100, title="$q(x)$", device="cpu", LOW = -4, HIGH = 4, plt_name = "Default"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True, sharex=True)
    axs[0].scatter(dataset[:, 0], dataset[:, 1], alpha=0.5, s=15)
    axs[1].scatter(gen_samples[:, 0], gen_samples[:, 1], alpha=0.5, s=15)
    axs[0].set_xlim([-4, 4])
    axs[0].set_ylim([-4, 4])
    plt.gcf().set_size_inches(5, 5)
    wandb.log({plt_name: wandb.Image(plt)})


def plot_kde_density_ax(
        ax, gen_samples, LOW = -4, HIGH = 4, title = None, cmap='magma',
        fontsize=18):
    kernel = stats.gaussian_kde(gen_samples.T, bw_method=0.1)
    X, Y = np.mgrid[LOW:HIGH:200j, LOW:HIGH:200j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.pcolormesh(X, Y, Z, cmap=cmap)
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.tick_params(bottom=False, labelbottom=False, top=False)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

def configure_plt(fontsize=8, poster=True):
    """Configure matplotlib with TeX and seaborn."""
    rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Computer Modern Roman']})
    usetex = matplotlib.checkdep_usetex(True)
    params = {'axes.labelsize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize - 2,
              'ytick.labelsize': fontsize - 2,
              'text.usetex': usetex,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette('colorblind')
    sns.set_style("ticks")
    if poster:
        sns.set_context("poster")
