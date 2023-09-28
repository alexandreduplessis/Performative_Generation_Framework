"""
Code taken from
https://github.com/rtqichen/residual-flows/blob/master/train_img.py#L207
"""

import argparse
import time
import math
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


from perfgen.utils import mix_data

def cifar_mix_dataloader(args, true_data, gen_data, random_state=0):
    perm = torch.randperm(true_data.size(0))
    n_old_samples = int(len(true_data) * args.prop_old)
    idx = perm[:n_old_samples]
    new_data = mix_data(true_data[idx], gen_data, random_state=random_state)
    new_train_dataset = torch.utils.data.TensorDataset(new_data)
    new_train_loader = torch.utils.data.DataLoader(
        new_train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers)
    return new_train_loader

def cifar_dataloader(
        args,
        dataroot="/network/datasets/cifar10.var/cifar10_torchvision/"):
    transform_train = transforms.Compose([
        # transforms.Resize(args.imagesize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(args.imagesize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # add_noise,
    ])
    train_dataset = datasets.CIFAR10(
        dataroot, train=True, transform=transform_train, download=False)
    # TODO adapt
    if args.prototype:
        idx = list(range(0, 1000))
        train_dataset= torch.utils.data.Subset(train_dataset, idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,  # For debuging
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            dataroot, train=False, transform=transform_test,
            download=False),
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )
    return train_loader, test_loader
