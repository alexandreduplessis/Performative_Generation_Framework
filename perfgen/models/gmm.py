import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state
from perfgen.utils import wasserstein_distance
import torch


class Gaussian_Mixture_Model():
    """
    Mixture of Gaussians

    Parameters
    ----------
    mus : array
        Means of the Gaussians
    sigmas : array
        Standard deviations of the Gaussians
    weights : array
        Weights of the Gaussians

    Methods
    -------
    train(data, epochs)
        Train the model (epochs is not used here)
        Returns the train loss (here not used)
    generate(nb_samples)
        Generate new data
    eval(data, **kwargs)
        Evaluate the model on data with given metrics
        Returns a dictionnary of metrics
    """
    def __init__(
            self, n_gaussians=3, dim=1, rng=check_random_state(0),mus=None,
            sigmas=None, weights=None):
        if mus is None:
            mus = np.array([range(n_gaussians)]*dim).reshape(n_gaussians, dim)
        if sigmas is None:
            sigmas = np.ones((n_gaussians, dim, dim))
        if weights is None:
            weights = np.array([1/n_gaussians]*n_gaussians)

        self.n_gaussians = n_gaussians
        self.dim = dim
        self.rng = rng
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights
        self.losses = []
        self.name = f'{self.dim}D Gaussian Mixture Model'
        self.metrics_titles = {'oldmean': 'Mean error', 'oldstd': 'Standard deviation error', 'oldwasserstein': 'Pseudo-Wasserstein distance',\
                                    'evalmean': 'Mean error', 'evalstd': 'Standard deviation error', 'evalwasserstein': 'Pseudo-Wasserstein distance'}

    def train(self, data, epochs):
        self.losses = []
        for epoch in range(epochs):
            gmm = GaussianMixture(
                n_components=self.n_gaussians, covariance_type='full',
                random_state=self.rng)
            gmm.fit(data)
            self.mus = gmm.means_
            self.sigmas = gmm.covariances_
            self.weights = gmm.weights_
            self.losses.append(-1.)
        self.losses = np.array(self.losses)
        return self.losses

    def get_theta(self):
        return {'mus': self.mus, 'sigmas': self.sigmas, 'weights': self.weights}

    def generate(self, nb_samples, save_path=None):
        samples = []
        for _ in range(nb_samples):
            i = self.rng.choice(self.n_gaussians, p=self.weights)
            new_sample = self.rng.multivariate_normal(self.mus[i], self.sigmas[i])
            samples.append(new_sample)
        samples = np.array(samples)
        if save_path is not None:
            # torch save
            torch.save(torch.from_numpy(samples), save_path)
        return samples

    def eval(self, data, **kwargs):
        metrics = {}
        # compute the std and mean of the data, taking weights into account
        data_gen = self.generate(len(data))
        data_std = np.std(data, axis=0)
        model_std = np.std(data_gen, axis=0)
        metrics['std'] = np.linalg.norm(model_std)
        metrics['wasserstein'] = wasserstein_distance(data, data_gen)
        return metrics

    def log_prob(self, data):
        # check the dimension of the data
        assert data.shape[1] == self.dim
        scores = []
        for sample in data:
            score = 0
            for i in range(self.n_gaussians):
                score += self.weights[i] * np.exp(-0.5 * np.dot(np.dot((sample - self.mus[i]), np.linalg.inv(self.sigmas[i])), (sample - self.mus[i]).T))
            scores.append(np.log(score))
        return torch.tensor(scores)

    def load(self, path):
        theta = np.load(path, allow_pickle=True).item()
        self.mus = theta['mus']
        self.sigmas = theta['sigmas']
        self.weights = theta['weights']
        self.n_gaussians, self.dim = self.mus.shape  # TODO to be fixed

    def save_model(self, path):
        np.save(path, self.get_theta())
