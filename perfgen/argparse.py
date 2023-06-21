import argparse
import time


def my_parser():
    parser = argparse.ArgumentParser(description='Performative Generator')
    parser.add_argument('--model', type=str, default='gmm', help='Model to train and generate data')
    parser.add_argument('--nb_iters', type=int, default=200, help='Number of iterations') # Warning: small values of nb_iters can be highly misleading when reading the results
    parser.add_argument('--nb_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--data', type=str, default='8gaussians', help='Dataset to use')
    parser.add_argument('--prop_old', type=float, default=0., help='Proportion of old data')
    parser.add_argument('--nb_new', type=int, default=100, help='Number of new datapoints to generate')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Frequency of checkpoints')
    parser.add_argument('--checkpoint_nb_gen', type=int, default=1000, help='Number of samples to generate at each checkpoint')
    parser.add_argument('--path', type=str, default="", help='Name of the experiment')

    args = parser.parse_args()
    if args.path == "":
        args.path = './checkpoints/' + args.model + '/' + args.data + '/' + str(args.nb_iters)
    else:
        args.path = args.path

    return args
