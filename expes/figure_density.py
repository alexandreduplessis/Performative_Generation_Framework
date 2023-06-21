import numpy as np
import torch
from matplotlib import cm
from perfgen.argparse import my_parser
from perfgen.models.gmm import Gaussian_Mixture_Model


def plt_density(
        model, ax, npts=100, memory=100, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    z = torch.tensor(z, requires_grad=False).type(torch.float32).to(device)

    with torch.no_grad:
        log_prob = model.log_prob(z)
        prob = torch.exp(log_prob).reshape(npts, npts)
    prob = prob.cpu().numpy()
    ax.imshow(prob, cmap=cm.magma)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

args = my_parser()
model = Gaussian_Mixture_Model()
model.load(args.path + '/' + "model_" + '0.pt')
