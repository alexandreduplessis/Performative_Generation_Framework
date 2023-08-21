import argparse, configparser
import time
import torch
import json
from omegaconf import DictConfig
from omegaconf import OmegaConf
import yaml


def my_parser():
    parser = argparse.ArgumentParser(description='Performative Generator')
    parser.add_argument('--model', type=str, default='bnaf', help='Model to train and generate data')
    parser.add_argument('--n_retrain', type=int, default=1, help='Number of iterations') # Warning: few retraining can be misleading when reading the results
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--data', type=str, default='8gaussians', help='Dataset to use')
    parser.add_argument('--prop_old', type=float, default=0., help='Proportion of old data')
    parser.add_argument('--nb_new', type=int, default=-1, help='Number of new datapoints to generate')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Frequency of checkpoints')
    parser.add_argument('--checkpoint_nb_gen', type=int, default=1000, help='Number of samples to generate at each checkpoint')
    parser.add_argument('--expe_name', type=str, default="", help='Name of the experiment')
    parser.add_argument('--cold_start', type=bool, default=False, help='Reset the model at each iteration')
    parser.add_argument('--n_epochs', type=int, default=-1, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--last_run', type=bool, default=False, help="Use the last experiment's arguments")
    parser.add_argument('--exp_name', type=str, default="", help='Name of the experiment')
    parser.add_argument("-c", "--config_file", type=str, default = "./expes/configs.conf", help='Config file')


    import ipdb; ipdb.set_trace()
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict = {k: v for k, v in args_dict.items() if v is not None}
    yaml_config = OmegaConf.load('base.yaml')
    merged_args_dict = OmegaConf.update(cfg=yaml_config, key=args_dict.keys())
    import ipdb; ipdb.set_trace()
    # merged_args_dict = OmegaConf.merge(init_config, args_dict)

    new_parser = generate_parser_from_dict(merged_args_dict)
    args = new_parser.parse_args()

    # if args.last_run:
    #     try:
    #         with open('./runs/last_run.json', 'r') as f:
    #             args_dict = json.load(f)
    #         args = argparse.Namespace(**args_dict)
    #     except:
    #         print("No last run found")
    #         exit(1)
    # else:
    #     # save the arguments in a json file in ./runs/last_run.json
    #     with open('./runs/last_run.json', 'w') as f:
    #         json.dump(args, f, indent=4)

    if args.exp_name == "":
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")

    # if args.device == 'None':
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


    # if args.dump_path == "":
    import ipdb; ipdb.set_trace()
    args.dump_path = './checkpoints/' + args.model + '/' + args.data + '/' + 'n_retrain_' + str(args.n_retrain) + '/' + 'n_samples_' + str(args.n_samples) + '/' + 'cold_start_' + str(args.cold_start) + '/' + 'prop_old_' + str(args.prop_old)
    # else:
    #     args.dump_path = args.expe_name

    return args


# import argparse
def generate_parser_from_dict(config_dict):
    parser = argparse.ArgumentParser(description="Generated from dictionary")
    for arg, attributes in config_dict.items():
        # import ipdb; ipdb.set_trace()
        parser.add_argument(f"--{arg}", type=type(attributes))
    return parser
