import numpy as np
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from perfgen.argparse import my_parser
from perfgen.models.gmm import Gaussian_Mixture_Model
from perfgen.models.flow import Normalizing_Flow
from tqdm import tqdm

LOW = -4
HIGH = 4

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
if args.model == 'gmm':
    model = Gaussian_Mixture_Model(
        n_gaussians=50, dim=2)
elif args.model == 'flow':
    model = Normalizing_Flow()
else:
    raise NotImplementedError
# TODO modify format, no more npy


n_plots = 10  # TODO read automatically
indices = np.arange(0, n_plots) * args.checkpoint_freq
assert len(indices) == n_plots
fig, axs = plt.subplots(1, n_plots, figsize=(10, 10))

for idx_arr, idx_checkpoint in enumerate(tqdm(indices)):
    model.load(args.path + '/' + "model_" + str(idx_checkpoint))
    plt_density(model, axs[idx_arr])
    axs[idx_arr].set_title("Epoch " + str(idx_checkpoint))

plt.show()