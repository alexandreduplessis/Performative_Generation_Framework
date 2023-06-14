import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from utils import mix_data
from models.gaussian1D import Gaussian_Estimator_1D
from models.gmm1D import Gaussian_Mixture_Model_1D
from models.gmm import Gaussian_Mixture_Model
from datasets import two_moons_dataset

class Performative_Generator():
    """
    Performative Generator

    Parameters
    ----------
    model : object
        Model to train and generate data
        model is assumed to have
            - a train method of the form model.train(data, epochs, **kwargs)
            - a generate method of the form model.generate(nb_samples, **kwargs)
            - an eval method of the form model.eval(data, **kwargs) that returns a dictionary of metrics
    data : array
        Old data
    nb_iters : int
        Number of iterations
    prop_old_schedule : array
        Proportion of old data at each iteration
    nb_new_schedule : array
        Number of new datapoints to generate at each iteration
    epochs_schedule : array
        Number of epochs to train the model at each iteration
    eval_schedule : array
        Iterations at which to evaluate the model
    eval_datas : array
        Dataset on which to evaluate the model
    """
    def __init__(self, model, data, nb_iters, prop_old_schedule, nb_new_schedule, epochs_schedule, eval_schedule, eval_data=None):
        self.model = model
        self.data = data
        self.old_data = data.copy()
        self.nb_iters = nb_iters
        self.prop_old_schedule = prop_old_schedule
        self.nb_new_schedule = nb_new_schedule
        self.epochs_schedule = epochs_schedule
        self.eval_schedule = eval_schedule
        self.eval_data = eval_data
    
    
    def train(self):
        metrics = {}
        metrics['indices'] = self.eval_schedule
        for i in tqdm(range(self.nb_iters)):
            # Generate data to train
            nb_old = int(self.prop_old_schedule[i] * self.data.shape[0])
            nb_new = self.nb_new_schedule[i]
            data_to_train = mix_data(self.old_data[:nb_old], self.model.generate(nb_new).reshape((nb_new, self.old_data.shape[1])))
            self.data = data_to_train.copy() # not useful

            # Train model
            losses = self.model.train(data_to_train, self.epochs_schedule[i])

            if i in self.eval_schedule:
                # Evaluate on old_data
                new_metrics = self.model.eval(self.old_data)
                if i == self.eval_schedule[0]:
                    for keys in new_metrics.keys():
                        metrics["old"+str(keys)] = np.array([new_metrics[keys]])
                else:
                    for keys in new_metrics.keys():
                        metrics["old"+str(keys)] = np.concatenate([metrics["old"+str(keys)], np.array([new_metrics[keys]])])
                if self.eval_data is not None:
                    # Evaluate on eval_data
                    new_metrics = self.model.eval(self.old_data)
                    if i == self.eval_schedule[0]:
                        for keys in new_metrics.keys():
                            metrics["eval"+str(keys)] = np.array([new_metrics[keys]])
                    else:
                        for keys in new_metrics.keys():
                            metrics["eval"+str(keys)] = np.concatenate([metrics["eval"+keys], np.array([new_metrics[keys]])])
                    
        return metrics
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performative Generator')
    parser.add_argument('--model', type=str, default='gmm', help='Model to train and generate data')
    parser.add_argument('--nb_iters', type=int, default=200, help='Number of iterations') # Warning: small values of nb_iters can be highly misleading when reading the results
    parser.add_argument('--nb_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--data', type=str, default='gaussians', help='Dataset to use')

    args = parser.parse_args()
    nb_samples = args.nb_samples
    if args.data == 'gaussians':
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], nb_samples)
        dim=2
    elif args.data == 'moons':
        data = two_moons_dataset(nb_samples=nb_samples, noise=.1)
        dim=2
    
    if args.model == 'gmm':
        model = Gaussian_Mixture_Model(nb=3, dim=dim)
    else:
        raise NotImplementedError
    
    nb_iters = args.nb_iters
    prop_old_schedule = np.array([1.] + [0] * nb_iters)
    # prop_old_schedule = np.ones(nb_iters)
    nb_new_schedule = [0] + [1000] * nb_iters
    # nb_new_schedule = np.array(range(nb_iters)) * 10
    epochs_schedule = [1] * nb_iters
    eval_schedule = np.arange(0, nb_iters, 1)

    performative_generator = Performative_Generator(model, data, nb_iters, prop_old_schedule, nb_new_schedule, epochs_schedule, eval_schedule)
    metrics = performative_generator.train()

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