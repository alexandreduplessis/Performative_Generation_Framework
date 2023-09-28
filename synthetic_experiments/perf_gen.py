import os
import torch
import numpy as np
from tqdm import tqdm

from utils import wasserstein_distance, plot_data, plot_density
from models.bnaf import BNAFlow

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
RESULTS_PATH = "results"


class PerfGenExperiment:
    model_name = ""
    eval_fn = wasserstein_distance

    n_gen_samples = 5000
    n_retrain = 25
    n_train_iter = 1000

    def __init__(
        self, dataset_name, train_data, prop_gen, retrain_initial=True
    ) -> None:
        self.dataset_name = dataset_name
        self.train_data = train_data
        self.prop_gen = prop_gen  # Set to -1 for only gen data
        self.retrain_initial = retrain_initial

        os.makedirs(self.get_path(), exist_ok=True)

        self.initialize_model()
        self.gen_datas = []
        self.evals = []

    def get_name(self):
        return f"{self.model_name}_{self.dataset_name}_{self.prop_gen}"

    def get_path(self):
        return os.path.join(
            RESULTS_PATH, self.model_name, self.dataset_name, str(self.prop_gen)
        )

    def get_model_path(self):
        return os.path.join(
            RESULTS_PATH, self.model_name, self.dataset_name, "initial_model.pt"
        )

    def generate(self, n_samples):
        pass

    def train(self, data):
        pass

    def mix(self, gen_data):
        if self.prop_gen == -1:
            return gen_data

        n_gen_samples = int(len(gen_data) * self.prop_gen)
        idx = np.random.choice(len(gen_data), n_gen_samples, replace=False)
        return torch.cat([self.train_data, gen_data[idx]], dim=0)

    def __call__(self):
        experiment_path = self.get_path()
        print(f"Running experiment for {self.get_name()}:")

        for iter in tqdm(range(self.n_retrain)):
            curr_path = os.path.join(experiment_path, str(iter))
            os.makedirs(curr_path, exist_ok=True)

            # Generate data and evaluate
            gen_data = self.generate(self.n_gen_samples)
            # eval_res = self.eval_fn(gen_data)
            self.gen_datas.append(gen_data)
            torch.save(gen_data, os.path.join(experiment_path, str(iter), "data.pt"))
            # torch.save(eval_res, os.path.join(experiment_path, str(iter), "eval.pt"))

            mixed_data = self.mix(gen_data)

            # Train model
            print(f"Training on {len(mixed_data)} samples.")
            self.train(mixed_data, self.n_train_iter)

            # Display iter results
            print(f"Retrain {iter}")
            plot_density(self.train_data, self.model)
