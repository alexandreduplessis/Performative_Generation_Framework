import numpy as np
import torch
import wandb
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from perfgen.argparse import my_parser
from perfgen.models.gmm import Gaussian_Mixture_Model
from perfgen.models.flow import Normalizing_Flow
from perfgen.models.bnaf import BNAFlow
from perfgen.models.simple_diffusion import SimpleDiffusion
from tqdm import tqdm

def plot_samples(
        gen_samples, ax, npts=100, memory=100, title="$q(x)$", device="cpu", LOW = -4, HIGH = 4):
    ax.scatter(gen_samples[:, 0], gen_samples[:, 1], alpha=0.5, s=15)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

# TODO: Model Forward Pass Should be on CUDA
def plt_density(
        model, ax, npts=100, memory=100, title="$q(x)$", device="cpu", LOW = -4, HIGH = 4):
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

wandb.login()

args = my_parser()
if args.model == 'gmm':
    n_gaussians = 8  # To be investigated
    model = Gaussian_Mixture_Model(
        n_gaussians=n_gaussians, dim=2)
elif args.model == 'flow':
    model = Normalizing_Flow()
elif args.model == 'bnaf':
    model = BNAFlow()
elif args.model == 'simplediff':
    model = SimpleDiffusion()
else:
    raise NotImplementedError

run = wandb.init(
    project="Performative_Generation_Framework",
    config={
        "n_retrain": args.n_retrain,
        "n_samples": args.n_samples,
        "data": args.data,
        "prop_old": args.prop_old,
        "nb_new": args.nb_new,
        "checkpoint_freq": args.checkpoint_freq,
        "checkpoint_nb_gen": args.checkpoint_nb_gen,
        "exp_path": args.dump_path,
        "model": args.model,
        "cold_start": args.cold_start
    },
    name=args.exp_name + "_viz"
    )

n_plots = 2  # TODO read automatically
indices = np.arange(0, n_plots) * args.checkpoint_freq
assert len(indices) == n_plots
fig, axs = plt.subplots(1, n_plots, figsize=(10, 10))

for idx_arr, idx_checkpoint in enumerate(tqdm(indices)):
    model.load(args.dump_path + '/' + "model_" + str(idx_checkpoint))
    gen_samples = model.generate(1000)
    print(gen_samples.std())
    if args.model == 'simplediff':
        plt_samples(gen_samples, axs[idx_arr])
    else:
        plt_density(model, axs[idx_arr])
    axs[idx_arr].set_title(str(idx_checkpoint))

# plt.savefig(args.dump_path + "/fig.pdf")
plt.show()
wandb.log({"fig": fig})
