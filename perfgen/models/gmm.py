import numpy as np
from sklearn.mixture import GaussianMixture
from perfgen.utils import wasserstein_distance


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
    def __init__(self, nb=3, dim=1, mus=None, sigmas=None, weights=None):
        if mus is None:
            mus = np.array([range(nb)]*dim).reshape(nb, dim)
        if sigmas is None:
            sigmas = np.ones((nb, dim, dim))
        if weights is None:
            weights = np.array([1/nb]*nb)

        self.nb = nb
        self.dim = dim
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
            gmm = GaussianMixture(n_components=self.nb, covariance_type='full')
            gmm.fit(data)
            self.mus = gmm.means_
            self.sigmas = gmm.covariances_
            self.weights = gmm.weights_
            self.losses.append(-1.)
        self.losses = np.array(self.losses)
        return self.losses

    def generate(self, nb_samples):
        samples = []
        for _ in range(nb_samples):
            i = np.random.choice(self.nb, p=self.weights)
            new_sample = np.random.multivariate_normal(self.mus[i], self.sigmas[i])
            samples.append(new_sample)
        samples = np.array(samples)
        return samples

    def eval(self, data, **kwargs):
        metrics = {}
        # compute the std and mean of the data, taking weights into account
        data_gen = self.generate(len(data))
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        model_mean = np.mean(data_gen, axis=0)
        model_std = np.std(data_gen, axis=0)
        metrics['mean'] = np.linalg.norm(data_mean - model_mean)
        metrics['std'] = np.linalg.norm(data_std - model_std)
        metrics['wasserstein'] = wasserstein_distance(data, data_gen)
        return metrics
