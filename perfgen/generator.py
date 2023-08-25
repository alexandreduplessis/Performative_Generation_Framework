import numpy as np
from tqdm import tqdm
import torch
import wandb


from perfgen.utils import mix_data
from perfgen.plot_density import plt_density, plot_kde_density
from perfgen.datasets.toy_data import sample_2d_data
from perfgen.datasets.cifar import cifar_mix_dataloader, cifar_dataloader

from torchvision.utils import save_image
from einops import rearrange

class Performative_Generator():
    """
    Performative Generator

    Parameters
    ----------
    model : object
        Model to train and generate data
        model is assumed to have
            - a train method of the form model.train(data, epochs, **kwargs)
            - a generate method of the form model.generate(n_samples, **kwargs)
            - an eval method of the form model.eval(data, **kwargs) that returns a dictionary of metrics
    n_retrain : int
        Number of iterations
    save_gen_samples : bool, default=False
    args : parser with the following arguments
        data : array
            Old data
        prop_old_schedule : array
            Proportion of old data at each iteration
        nb_new_schedule : array
            Number of new datapoints to generate at each iteration
        eval_schedule : array
            Iterations at which to evaluate the model
        eval_datas : array
            Dataset on which to evaluate the model
    """
    def __init__(
        self, args, model, n_retrain):
        self.args = args
        self.model = model
        self.dataname = args.dataname
        if self.dataname != "cifar":
            data = sample_2d_data(self.dataname, args.n_samples)
            # TODO add rng back in sample_2d_data
            self.data = data
            self.init_data = data.clone()
        else:
            self.init_train_loader, self.test_loader = cifar_dataloader(args)
            self.init_data = torch.cat(
                [data for data, _ in self.init_train_loader], dim=0)

        self.n_retrain = n_retrain
        self.prop_old_schedule = args.prop_old_schedule
        self.nb_new_schedule = args.nb_new_schedule
        self.n_epochs = args.n_epochs
        self.n_finetune_epochs = args.n_finetune_epochs
        self.eval_schedule = args.eval_schedule
        self.eval_data = args.eval_data
        self.checkpoint_freq = args.checkpoint_freq
        self.checkpoint_nb_gen = args.checkpoint_nb_gen
        self.dump_path = args.dump_path
        self.cold_start = args.cold_start
        self.device = args.device
        self.save_gen_samples = args.save_gen_samples



    def train(self):
        metrics = {}
        metrics['indices'] = self.eval_schedule
        # TODO do iteration outside the loop
        for i in tqdm(range(self.n_retrain)):
            print("n_retrain % i" % i)
            if self.cold_start:
                self.model.cold_start()
            nb_new = self.nb_new_schedule[i]
            if self.dataname != "cifar":
                # Generate data to train
                nb_old = int(self.prop_old_schedule[i] * self.data.shape[0])
                train_loader = mix_data(
                    self.init_data[:nb_old],
                    self.model.generate(nb_new).reshape((nb_new, self.init_data.shape[1])).cpu())
                self.data = train_loader.clone()
            else:
                # Everything should be done on cpu here
                if i == 0:
                    train_loader = self.init_train_loader
                    n_epochs = self.n_epochs
                else:
                    train_loader = cifar_mix_dataloader(
                        self.args,
                        self.init_data.cpu(),
                        gen_data.cpu())
                    n_epochs = self.n_finetune_epochs

            # Train model
            self.model.train(train_loader, n_epochs)
            # Generate data
            # TODO put gen_gen on cpu directly
            # TODO batch this operation
            gen_data = self.model.generate(
                self.checkpoint_nb_gen, f"{self.dump_path}/generated_{i}.pt")
            # Evaluate the trained model
            self.eval_model(i, gen_data)
            self.plot_data(i, gen_data)
            self.save_model_and_data(i, gen_data)

        # One last save
        self.model.save_model(f"{self.dump_path}/model_final")
        gen_data = self.model.generate(self.checkpoint_nb_gen, f"{self.dump_path}/generated_final.pt")

        return metrics


    def eval_model(self, i, gen_data):
        if self.dataname != "cifar":
            metrics = self.model.eval(self.init_data)
        else:
            metrics = self.model.eval(
                self.init_train_loader, self.test_loader, gen_data.to('cpu'))
        for keys in metrics.keys():
            wandb.log({"eval"+str(keys): metrics[keys]})

    def save_model_and_data(self, i, gen_data):
        if i % self.checkpoint_freq == 0:
            self.model.save_model(
                f"{self.dump_path}/model_{i}")
            if (self.dump_path is not None) and self.save_gen_samples:
                tot_dump_path = f"{self.dump_path}/generated_samples_{i}.pt"
                if self.dataname != "cifar":
                    # Save the generated samples
                    torch.save(
                        gen_data.detach().cpu(), tot_dump_path)
                else:
                    pass
                    # TODO save by batch the samples



    def plot_data(self, i, gen_data):
        if self.dataname != "cifar":
            if ('Diff' in str(self.model)):
                plot_kde_density(
                    self.init_data, gen_data.cpu().numpy(), plt_name=f"density_{i}.png")
            else:
                plt_density(self.model, plt_name=f"density_{i}.png")
        else:
            tot_dump_path = f"{self.dump_path}/generated_samples_{i}.jpg"
            save_image(
                gen_data[:128, :].cpu().float(),
                tot_dump_path, nrow=16, padding=2)
            wandb.log({"Generated Data at retraining %i" %i: wandb.Image(tot_dump_path)})
