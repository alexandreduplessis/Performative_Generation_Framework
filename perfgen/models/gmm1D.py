import numpy as np
from sklearn.mixture import GaussianMixture

class Gaussian_Mixture_Model_1D():
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
    def __init__(self, nb=3, mus=None, sigmas=None, weights=None):
        if mus is None:
            mus = np.array(range(nb))
        if sigmas is None:
            sigmas = np.array([1]*nb)
        if weights is None:
            weights = np.array([1/nb]*nb)
        
        self.nb = nb
        self.mus = mus
        self.sigmas = sigmas
        self.weights = weights
        self.losses = []
        self.name = '1D Gaussian Mixture Model'
        self.metrics_titles = {'oldmean': 'Mean', 'oldstd': 'Standard deviation', 'oldweisserstein': 'Pseudo-Weisserstein distance',\
                                    'evalmean': 'Mean', 'evalstd': 'Standard deviation', 'evalweisserstein': 'Pseudo-Weisserstein distance'}
        
    def train(self, data, epochs):
        self.losses = []
        for epoch in range(epochs):
            gmm = GaussianMixture(n_components=self.nb)
            gmm.fit(data.reshape(-1, 1))
            self.mus = gmm.means_
            self.sigmas = np.sqrt(gmm.covariances_)
            self.weights = gmm.weights_
            self.losses.append(-1.)
        self.losses = np.array(self.losses)
        return self.losses
    
    def generate(self, nb_samples):
        samples = []
        for _ in range(nb_samples):
            i = np.random.choice(self.nb, p=self.weights)
            new_sample = np.random.normal(self.mus[i], self.sigmas[i])
            samples.append(new_sample.item())
        return np.array(samples)
    
    def eval(self, data, **kwargs):
        metrics = {}
        # metrics['mean'] = np.abs(np.mean(data) - np.mean(self.mus))
        # metrics['std'] = np.abs(np.std(data, ddof=1) - np.mean(self.sigmas))
        metrics['mean'] = np.mean(self.mus)
        metrics['std'] = np.mean(self.sigmas)
        metrics['weisserstein'] = np.linalg.norm(np.mean(data) - np.mean(self.mus)) + np.linalg.norm(np.std(data, ddof=1) - np.mean(self.sigmas))
        return metrics