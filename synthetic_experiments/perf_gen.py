import os
import torch
import numpy as np
from tqdm import tqdm


from utils import wasserstein_distance
from models.bnaf import BNAFlow

RESULTS_PATH = "results"

class PerfGen:
    model_name = ""
    model = None

    eval_fn = wasserstein_distance

    n_gen_samples = 1000
    n_retrain = 10

    def __init__(self, dataset_name, train_data, prop_gen) -> None:
        self.dataset_name = dataset_name
        self.train_data = train_data
        self.prop_gen = prop_gen

    def get_name(self):
        return f"{self.model_name}_{self.dataset_name}_{self.prop_gen}"
    
    def get_path(self):
        return os.path.join(RESULTS_PATH, self.model_name, self.dataset_name, str(self.prop_gen))
    
    def generate(self, n_samples):
        pass

    def train(self, data):
        pass

    def mix(self, gen_data):
        n_gen_samples = int(len(gen_data) * self.prop_gen)
        idx = np.random.choice(len(gen_data), n_gen_samples, replace=False)
        return torch.cat([self.train_data, gen_data[idx]], dim=0)

    def __call__(self):
        experiment_path = self.get_path()
        os.makedirs(experiment_path, exist_ok=True)
        print(f"Running experiment for {self.get_name()}:")

        for iter in tqdm(range(self.n_retrain)):
            curr_path = os.path.join(experiment_path, str(iter))
            os.makedirs(curr_path, exist_ok=True)

            # Generate data and evaluate
            gen_data = self.generate(self.n_gen_samples)
            # eval_res = self.eval_fn(gen_data)
            torch.save(gen_data, os.path.join(experiment_path, str(iter), "data.pt"))
            # torch.save(eval_res, os.path.join(experiment_path, str(iter), "eval.pt"))

            mixed_data = self.mix(gen_data)

            # Train model
            print(f"Training on {len(mixed_data)} samples.")  
            self.train(mixed_data)

class BNAFPerfGen(PerfGen):
    model_name = "BNAF"
    model = BNAFlow()

    def generate(self, n_samples):
        return self.model.generate(n_samples)
        
    def train(self, data):
        self.model.train(data)

