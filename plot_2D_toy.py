import torch
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.utils import check_random_state

from perfgen.argparse import generate_parser_from_dict
from perfgen.argparse import get_dump_path_from_yaml_path
from perfgen.plot_density import configure_plt, plot_kde_density_ax
from perfgen.datasets.toy_data import sample_2d_data

path_to_collapse_yaml = 'configs/2D_toy_collapse.yaml'
path_to_stable_yaml = 'configs/2D_toy_stable.yaml'

dump_path_collapse = get_dump_path_from_yaml_path(path_to_collapse_yaml)
dump_path_stable = get_dump_path_from_yaml_path(path_to_stable_yaml)

rng = check_random_state(0)
true_samples = sample_2d_data('8gaussians', 5_000, rng)


configure_plt()
n_retrains = [0, 10, 20, 50]
n_subplots = len(n_retrains)

fontsize= 18
fig, axs = plt.subplots(2, n_subplots + 1, figsize=(12, 4), sharey=True, sharex=True)


plot_kde_density_ax(
    axs[0, 0], true_samples, title = 'Ground Truth', fontsize=fontsize)
plot_kde_density_ax(axs[1, 0], true_samples)

for idx_ax, n_retrain in enumerate(n_retrains):


    gen_samples = torch.load(
        dump_path_collapse + '/generated_samples_%i.pt' % n_retrain)
    if idx_ax == 0:
        title = 'No retraining'
    else:
        title = '$\# \mathrm{retrain.} = %i$' % n_retrain
    plot_kde_density_ax(
        axs[0, idx_ax + 1], gen_samples, title = title)

    gen_samples = torch.load(
        dump_path_stable + '/generated_samples_%i.pt' % n_retrain)
    plot_kde_density_ax(axs[1, idx_ax + 1], gen_samples)

axs[0, 0].set_ylabel('Full \n Replacement', fontsize=fontsize)
axs[1, 0].set_ylabel('Partial \n Replacement', fontsize=fontsize)



fig_dir = "../perfgen_tex/figures/"
fig_dir_svg = "../perfgen_tex/figures_svg/"
# save_fig = False
save_fig = True
plt.tight_layout()
if save_fig:
    fig_name = "2D_toy"
    fig.savefig(fig_dir + fig_name + ".pdf", bbox_inches="tight")
    fig.savefig(fig_dir_svg + fig_name + ".svg", bbox_inches="tight")
    # fig.savefig(fig_dir + fig_name + ".svg", bbox_inches="tight")
    # legend = ax[1, -1].legend(bbox_to_anchor=(1, 0.5), ncol=3)
    # export_legend(legend, filename=fig_dir + fig_name + "_legend.pdf")
    # ax[1, -1].get_legend().remove()
plt.show(block=False)
