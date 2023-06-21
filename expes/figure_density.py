import numpy as np
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from perfgen.argparse import my_parser
from perfgen.models.gmm import Gaussian_Mixture_Model

LOW = -8
HIGH = 8

def plt_density(
        model, ax, npts=100, memory=100, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    z = torch.tensor(z, requires_grad=False).type(torch.float32).to(device)

    with torch.no_grad():
        log_prob = model.log_prob(z)
        prob = torch.exp(log_prob).reshape(npts, npts)
    prob = prob.cpu().numpy()
    # ax.imshow(prob)
    ax.imshow(prob, cmap=cm.magma)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

args = my_parser()
model = Gaussian_Mixture_Model()
# TODO modify format, no more npy

n_plots = 5  # TODO read automatically
fig, axs = plt.subplots(1, n_plots, figsize=(10, 10))

for idx_arr, idx_checkpoint in enumerate(
        np.arange(n_plots) * args.checkpoint_freq):
    model.load(args.path + '/' + "model_" + str(idx_checkpoint) + '.npy')

    plt_density(model, axs[idx_arr])

plt.show()
