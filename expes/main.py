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
from perfgen.generator import Performative_Generator
from perfgen.datasets.toy_data import sample_2d_data
from perfgen.argparse import my_parser


if __name__ == "__main__":
    date_str = time.strftime("%Y%m%d-%H%M%S")
    args = my_parser()
    nb_samples = args.nb_samples

    rng = check_random_state(0)
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    info = {}
    info['date'] = date_str
    info['nb_iters'] = args.nb_iters
    info['nb_samples'] = args.nb_samples
    info['data'] = args.data
    info['prop_old'] = args.prop_old
    info['nb_new'] = args.nb_new
    info['checkpoint_freq'] = args.checkpoint_freq
    info['checkpoint_nb_gen'] = args.checkpoint_nb_gen
    info['exp_name'] = args.path
    info['model'] = args.model

    np.save(args.path + '/info.npy', info)

    data = sample_2d_data(args.data, nb_samples, rng)
    if args.model == 'gmm':
        model = Gaussian_Mixture_Model(
            n_gaussians=50, dim=2, rng=rng)
    else:
        raise NotImplementedError

    nb_iters = args.nb_iters
    prop_old_schedule = np.array([1.] + [args.prop_old] * nb_iters)
    nb_new_schedule = [0] + [args.nb_new] * nb_iters
    epochs_schedule = [1] * nb_iters
    eval_schedule = np.arange(0, nb_iters, 1)

    performative_generator = Performative_Generator(model=model, data=data, nb_iters=nb_iters, prop_old_schedule=prop_old_schedule, nb_new_schedule=nb_new_schedule, epochs_schedule=epochs_schedule, eval_schedule=eval_schedule, checkpoint_freq=args.checkpoint_freq, checkpoint_nb_gen=args.checkpoint_nb_gen, exp_name=args.path)
    metrics, theta_list = performative_generator.train()

    np.save(args.path + '/metrics.npy', metrics)

    keys = list(metrics.keys())
    keys.remove('indices')
    keys_names = model.metrics_titles

    nb_plots = len(keys)

    # make subplots
    fig, axs = plt.subplots(nb_plots, 1, figsize=(10, 10))
    for i, key in enumerate(keys):
        axs[i].plot(metrics['indices'], metrics[key])
        axs[i].set_title(f"{keys_names[key]} for {model.name}")
    plt.show()