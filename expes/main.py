import numpy as np
import torch
import os
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
from sklearn.utils import check_random_state
from perfgen.models.gmm import Gaussian_Mixture_Model
from perfgen.models.flow import Normalizing_Flow
from perfgen.models.simple_diffusion import SimpleDiffusion
from perfgen.generator import Performative_Generator
from perfgen.models.bnaf import BNAFlow
from perfgen.datasets.toy_data import sample_2d_data
from perfgen.argparse import my_parser

def main():
    wandb.login()
    date_str = time.strftime("%Y%m%d-%H%M%S")
    args = my_parser()
    n_samples = args.n_samples

    rng = check_random_state(0)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    info = {}
    info['date'] = date_str
    info['n_retrain'] = args.n_retrain
    info['n_samples'] = args.n_samples
    info['data'] = args.data
    info['prop_old'] = args.prop_old
    info['nb_new'] = args.nb_new
    info['checkpoint_freq'] = args.checkpoint_freq
    info['checkpoint_nb_gen'] = args.checkpoint_nb_gen
    info['dump_path'] = args.dump_path
    info['model'] = args.model
    info['cold_start'] = args.cold_start

    np.save(args.dump_path + '/info.npy', info)

    data = sample_2d_data(args.data, n_samples, rng)
    n_retrain = args.n_retrain
    if args.model == 'gmm':
        model = Gaussian_Mixture_Model(
            n_gaussians=50, dim=2, rng=rng)
    elif args.model == 'flow':
        model = Normalizing_Flow(args.device)
    elif args.model == 'bnaf':
        model = BNAFlow(args.device)
    elif args.model == 'simplediff':
        model = SimpleDiffusion(args.device)
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
            "dump_path": args.dump_path,
            "model": args.model,
            "cold_start": args.cold_start
        },
        name=args.exp_name
    )


    prop_old_schedule = np.array([1.] + [args.prop_old] * n_retrain)
    nb_new_schedule = [0] + [args.nb_new] * n_retrain
    eval_schedule = np.arange(0, n_retrain, 1)
    epochs_schedule = [args.n_epochs] * n_retrain

    performative_generator = Performative_Generator(
        model=model, data=data, n_retrain=n_retrain, prop_old_schedule=prop_old_schedule, nb_new_schedule=nb_new_schedule, epochs_schedule=epochs_schedule, eval_schedule=eval_schedule, checkpoint_freq=args.checkpoint_freq, checkpoint_nb_gen=args.checkpoint_nb_gen, dump_path=args.dump_path, cold_start=args.cold_start, device = args.device, save_gen_samples=True)
    metrics = performative_generator.train()

    np.save(args.dump_path + '/metrics.npy', metrics)
    np.save(args.dump_path + '/metrics.npy', metrics)


if __name__ == "__main__":
    main()
