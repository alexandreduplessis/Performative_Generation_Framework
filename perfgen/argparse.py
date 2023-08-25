import argparse
import time
import torch
from omegaconf import OmegaConf


def my_parser(path_to_yaml='configs/cifar_debug.yaml'):
    # def my_parser(path_to_yaml='configs/2D_toy_collapse.yaml'):
    yaml_config = OmegaConf.load(path_to_yaml)
    parser = generate_parser_from_dict(yaml_config)
    parser.add_argument(
        '--path_to_yaml', type=str, default='configs/base.yaml')
    args = parser.parse_args()

    if args.exp_name == "":
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.nb_new == -1:
        args.nb_new = args.n_samples

    if args.n_epochs == -1:
        if args.model == 'gmm':
            args.n_epochs = 1
        elif args.model == 'bnaf':
            args.n_epochs = 1_000
            # args.n_epochs = 20_000
        elif args.model == 'flow':
            args.n_epochs = 100
        elif args.model == 'simplediff':
            args.n_epochs = 200

    args.dump_path = get_dump_path(args)

    return args

def generate_parser_from_dict(config_dict):
    parser = argparse.ArgumentParser(description="Generated from dictionary")
    for arg, attributes in config_dict.items():
        parser.add_argument(
            f"--{arg}", type=type(attributes), default=attributes)
    return parser

def get_dump_path(args, cluster=True):
    dump_path = 'checkpoints/' + args.model + '/' + args.dataname + '/' + 'n_retrain_' + str(args.n_retrain) + '/' + 'n_samples_' + str(args.n_samples) + '/' + 'cold_start_' + str(args.cold_start) + '/' + 'prop_old_' + str(args.prop_old)
    if cluster:
        dump_path = './' + dump_path
    return dump_path

def get_dump_path_from_yaml_path(path_to_yaml, cluster=True):
    yaml_config = OmegaConf.load(path_to_yaml)
    parser = generate_parser_from_dict(yaml_config)
    parser.add_argument(
        '--path_to_yaml', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    dump_path = get_dump_path(args, cluster=False)
    return dump_path
