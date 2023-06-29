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