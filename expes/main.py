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
from perfgen.models.ddpm import DDPM
from perfgen.generator import Performative_Generator
from perfgen.models.bnaf import BNAFlow
from perfgen.argparse import my_parser
import sys

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
    info['n_epochs'] = args.n_epochs
    info['n_finetune_epochs'] = args.n_finetune_epochs
    info['n_samples'] = args.n_samples
    info['data'] = args.dataname
    info['prop_old'] = args.prop_old
    info['nb_new'] = args.nb_new
    info['checkpoint_freq'] = args.checkpoint_freq
    info['n_gen_samples'] = args.n_gen_samples
    info['dump_path'] = args.dump_path
    info['model'] = args.model
    info['cold_start'] = args.cold_start

    np.save(args.dump_path + '/info.npy', info)

    if args.model == 'gmm':
        model = Gaussian_Mixture_Model(
            n_gaussians=50, dim=2, rng=rng)
    elif args.model == 'flow':
        model = Normalizing_Flow(args.device)
    elif args.model == 'bnaf':
        model = BNAFlow(args.device)
    elif args.model == 'simplediff':
        model = SimpleDiffusion(args.device)
    elif args.model == 'ddpm':
        model = DDPM(
            args.device, args.dim, num_timesteps=args.num_timesteps,
            sampling_timesteps=args.sampling_timesteps)
    else:
        raise NotImplementedError

    run = wandb.init(
        project="Performative_Generation_Framework",
        config={
            "n_retrain": args.n_retrain,
            "n_samples": args.n_samples,
            "data": args.dataname,
            "prop_old": args.prop_old,
            "nb_new": args.nb_new,
            "checkpoint_freq": args.checkpoint_freq,
            "n_gen_samples": args.n_gen_samples,
            "dump_path": args.dump_path,
            "model": args.model,
            "cold_start": args.cold_start
        },
        name=args.exp_name
    )


    # This is dirty: we should change it
    n_retrain = args.n_retrain
    args.prop_old_schedule = np.array([1.] + [args.prop_old] * n_retrain)
    args.nb_new_schedule = [0] + [args.nb_new] * n_retrain
    args.eval_schedule = np.arange(0, n_retrain, 1)
    args.eval_data = None


    performative_generator = Performative_Generator(args, model=model)
    metrics = performative_generator.train()

    np.save(args.dump_path + '/metrics.npy', metrics)
    np.save(args.dump_path + '/metrics.npy', metrics)


if __name__ == "__main__":
    main()
