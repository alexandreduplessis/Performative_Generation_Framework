import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
from sklearn.utils import check_random_state
from perfgen.models.gaussian1D import Gaussian_Estimator_1D
from perfgen.models.gmm1D import Gaussian_Mixture_Model_1D
from perfgen.models.gmm import Gaussian_Mixture_Model
from datasets import two_moons_dataset
from perfgen.generator import Performative_Generator
from perfgen.datasets.toy_data import sample_2d_data



if __name__ == "__main__":
    date_str = time.strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser(description='Performative Generator')
    parser.add_argument('--model', type=str, default='gmm', help='Model to train and generate data')
    parser.add_argument('--nb_iters', type=int, default=200, help='Number of iterations') # Warning: small values of nb_iters can be highly misleading when reading the results
    parser.add_argument('--nb_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--data', type=str, default='8gaussians', help='Dataset to use')
    parser.add_argument('--prop_old', type=float, default=0., help='Proportion of old data')
    parser.add_argument('--nb_new', type=int, default=100, help='Number of new datapoints to generate')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Frequency of checkpoints')
    parser.add_argument('--checkpoint_nb_gen', type=int, default=1000, help='Number of samples to generate at each checkpoint')
    parser.add_argument('--exp_name', type=str, default=date_str, help='Name of the experiment')

    args = parser.parse_args()
    nb_samples = args.nb_samples

    rng = check_random_state(0)

    # create folder in ./checkpoints
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./checkpoints/' + args.exp_name):
        os.mkdir('./checkpoints/' + args.exp_name)

    data = sample_2d_data(args.data, nb_samples, rng)
    if args.model == 'gmm':
        model = Gaussian_Mixture_Model(nb=8, dim=2, rng=rng)
    else:
        raise NotImplementedError

    nb_iters = args.nb_iters
    prop_old_schedule = np.array([1.] + [args.prop_old] * nb_iters)
    nb_new_schedule = [0] + [args.nb_new] * nb_iters
    epochs_schedule = [1] * nb_iters
    eval_schedule = np.arange(0, nb_iters, 1)

    performative_generator = Performative_Generator(model=model, data=data, nb_iters=nb_iters, prop_old_schedule=prop_old_schedule, nb_new_schedule=nb_new_schedule, epochs_schedule=epochs_schedule, eval_schedule=eval_schedule, checkpoint_freq=args.checkpoint_freq, checkpoint_nb_gen=args.checkpoint_nb_gen, exp_name=args.exp_name)
    metrics, theta_list = performative_generator.train()

    keys = list(metrics.keys())
    keys.remove('indices')
    keys_names = model.metrics_titles

    nb_plots = len(keys)

    # make subplots
    # fig, axs = plt.subplots(nb_plots, 1, figsize=(10, 10))
    # for i, key in enumerate(keys):
    #     axs[i].plot(metrics['indices'], metrics[key])
    #     axs[i].set_title(f"{keys_names[key]} for {model.name}")
    # plt.show()

    # if len(theta_list.keys()) > 0:
    #     nb_plots = 10
    #     fig, axs = plt.subplots(1, nb_plots, figsize=(20, 3))
    #     keys = list(theta_list.keys())
    #     plot_keys = keys[::len(keys) // nb_plots]
    #     if len(plot_keys) > nb_plots:
    #         plot_keys = plot_keys[:nb_plots]
    #     for plot_id in range(nb_plots):
    #         key = keys[plot_keys[plot_id]]
    #         all = theta_list[key]
    #         model = Gaussian_Mixture_Model(nb=8, dim=2)
    #         model.load(all)
    #         X, Y = np.meshgrid(np.linspace(-6, 6), np.linspace(-6,6))
    #         XX = np.array([X.ravel(), Y.ravel()]).T
    #         Z = model.score_samples(XX)
    #         Z = Z.reshape((50,50))
    #         axs[plot_id].contour(X, Y, Z)
    #     plt.show()
