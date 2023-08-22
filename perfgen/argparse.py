import argparse
import time
import torch
from omegaconf import OmegaConf


def my_parser():
    yaml_config = OmegaConf.load('base.yaml')
    parser = generate_parser_from_dict(yaml_config)
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

    args.dump_path = './checkpoints/' + args.model + '/' + args.data + '/' + 'n_retrain_' + str(args.n_retrain) + '/' + 'n_samples_' + str(args.n_samples) + '/' + 'cold_start_' + str(args.cold_start) + '/' + 'prop_old_' + str(args.prop_old)

    return args

def generate_parser_from_dict(config_dict):
    parser = argparse.ArgumentParser(description="Generated from dictionary")
    for arg, attributes in config_dict.items():
        parser.add_argument(
            f"--{arg}", type=type(attributes), default=attributes)
    return parser
