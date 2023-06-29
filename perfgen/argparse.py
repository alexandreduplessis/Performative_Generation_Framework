import argparse, configparser
import time
import torch
import json


def my_parser():
    parser = argparse.ArgumentParser(description='Performative Generator')
    parser.add_argument('--model', type=str, default='bnaf', help='Model to train and generate data')
    parser.add_argument('--nb_iters', type=int, default=1, help='Number of iterations') # Warning: small values of nb_iters can be highly misleading when reading the results
    parser.add_argument('--nb_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--data', type=str, default='8gaussians', help='Dataset to use')
    parser.add_argument('--prop_old', type=float, default=0., help='Proportion of old data')
    parser.add_argument('--nb_new', type=int, default=-1, help='Number of new datapoints to generate')
    parser.add_argument('--checkpoint_freq', type=int, default=1, help='Frequency of checkpoints')
    parser.add_argument('--checkpoint_nb_gen', type=int, default=1000, help='Number of samples to generate at each checkpoint')
    parser.add_argument('--path', type=str, default="", help='Name of the experiment')
    parser.add_argument('--reset', type=bool, default=False, help='Reset the model at each iteration')
    parser.add_argument('--epochs', type=int, default=-1, help='Number of epochs')
    parser.add_argument('--device', type=str, default='None', help='Device to use')
    parser.add_argument('--last_run', type=bool, default=False, help="Use the last experiment's arguments")
    parser.add_argument('--exp_name', type=str, default="", help='Name of the experiment')
    parser.add_argument("-c", "--config_file", type=str, default = "./expes/configs.conf", help='Config file')

    args = parser.parse_args()

    # if args.config_file:
    #     config = configparser.ConfigParser()
    #     config.read(args.config_file)
    #     defaults = {}
    #     defaults.update(dict(config.items("Defaults")))
    #     parser.set_defaults(**defaults)
    #     args = parser.parse_args()

    if args.last_run:
        try:
            with open('./runs/last_run.json', 'r') as f:
                args_dict = json.load(f)
            args = argparse.Namespace(**args_dict)
        except:
            print("No last run found")
            exit(1)
    else:
        # save the arguments in a json file in ./runs/last_run.json
        args_dict = vars(args)
        with open('./runs/last_run.json', 'w') as f:
            json.dump(args_dict, f, indent=4)

    if args.exp_name == "":
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")

    if args.device == 'None':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.nb_new == -1:
        args.nb_new = args.nb_samples

    if args.epochs == -1:
        if args.model == 'gmm':
            args.epochs = 1
        elif args.model == 'bnaf':
            args.epochs = 1_000
            # args.epochs = 20_000
        elif args.model == 'flow':
            args.epochs = 100


    if args.path == "":
        args.path = './checkpoints/' + args.model + '/' + args.data + '/' + str(args.nb_iters) + '/' + str(args.nb_samples) + '/' + str(args.reset) + '/' + str(args.prop_old)
    else:
        args.path = args.path

    return args
