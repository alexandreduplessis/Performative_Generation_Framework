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
    # train_loader, test_loader = cifar_dataloader(args)
    new_data = mix_data(true_data, gen_data, random_state=random_state)
    new_train_dataset = torch.utils.data.TensorDataset(new_data)
    new_train_loader = torch.utils.data.DataLoader(
        new_train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers)
    return new_train_loader

def cifar_dataloader(args):
    transform_train = transforms.Compose([
        # transforms.Resize(args.imagesize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # add_noise,
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(args.imagesize),
        transforms.ToTensor(),
        # add_noise,
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.dataroot, train=True, transform=transform_train,
            download=False),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.nworkers,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.dataroot, train=False, transform=transform_test,
            download=False),
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.nworkers,
    )
    return train_loader, test_loader
