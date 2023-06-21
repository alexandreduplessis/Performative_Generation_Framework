import numpy as np
from perfgen.utils import mix_data
from tqdm import tqdm
import torch

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
    def __init__(self, model, data, nb_iters, prop_old_schedule, nb_new_schedule, epochs_schedule, eval_schedule, checkpoint_freq, checkpoint_nb_gen, exp_name, eval_data=None):
        self.model = model
        self.data = data
        self.old_data = data.copy()
        self.nb_iters = nb_iters
        self.prop_old_schedule = prop_old_schedule
        self.nb_new_schedule = nb_new_schedule
        self.epochs_schedule = epochs_schedule
        self.eval_schedule = eval_schedule
        self.eval_data = eval_data
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_nb_gen = checkpoint_nb_gen
        self.exp_name = exp_name


    def train(self):
        metrics = {}
        metrics['indices'] = self.eval_schedule
        theta = {}
        for i in tqdm(range(self.nb_iters)):
            # Generate data to train
            nb_old = int(self.prop_old_schedule[i] * self.data.shape[0])
            nb_new = self.nb_new_schedule[i]
            data_to_train = mix_data(self.old_data[:nb_old], self.model.generate(nb_new).reshape((nb_new, self.old_data.shape[1])))
            self.data = data_to_train.copy() # not useful

            # Train model
            losses = self.model.train(data_to_train, self.epochs_schedule[i])
            # Save model
            if i % self.checkpoint_freq == 0:
                self.model.save_model(f"./checkpoints/{self.exp_name}/model_{i}.pt")
                # generate
                gen_data = self.model.generate(self.checkpoint_nb_gen, f"./checkpoints/{self.exp_name}/generated_{i}.pt")
        # One last save
        self.model.save_model(f"./checkpoints/{self.exp_name}/model_final.pt")
        gen_data = self.model.generate(self.checkpoint_nb_gen, f"./checkpoints/{self.exp_name}/generated_final.pt")
            # if i in self.eval_schedule:
            #     theta[i] = self.model.get_theta()
            #     # Evaluate on old_data
            #     new_metrics = self.model.eval(self.old_data)
            #     if i == self.eval_schedule[0]:
            #         for keys in new_metrics.keys():
            #             metrics["old"+str(keys)] = np.array([new_metrics[keys]])
            #     else:
            #         for keys in new_metrics.keys():
            #             metrics["old"+str(keys)] = np.concatenate([metrics["old"+str(keys)], np.array([new_metrics[keys]])])
            #     if self.eval_data is not None:
            #         # Evaluate on eval_data
            #         new_metrics = self.model.eval(self.old_data)
            #         if i == self.eval_schedule[0]:
            #             for keys in new_metrics.keys():
            #                 metrics["eval"+str(keys)] = np.array([new_metrics[keys]])
            #         else:
            #             for keys in new_metrics.keys():
            #                 metrics["eval"+str(keys)] = np.concatenate([metrics["eval"+keys], np.array([new_metrics[keys]])])

        return metrics, theta
