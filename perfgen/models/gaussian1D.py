import numpy as np

class Gaussian_Estimator_1D():
    """
    Gaussian Estimator

    Parameters
    ----------
    mu : float
        Mean of the Gaussian
    sigma : float
        Standard deviation of the Gaussian

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
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.loss = []
        self.name = '1D Gaussian Estimator'
        self.metrics_titles = {'oldmean': 'Mean', 'oldstd': 'Standard deviation', 'oldweisserstein': 'Weisserstein distance',\
                                 'evalmean': 'Mean', 'evalstd': 'Standard deviation', 'evalweisserstein': 'Weisserstein distance'}

    def train(self, data, epochs):
        self.losses = []
        for epoch in range(epochs):
            self.mu = np.mean(data)
            self.sigma = np.std(data, ddof=1)
            self.losses.append(-1.)
        self.losse = np.array(self.losses)
        return self.losses

    def generate(self, nb_samples):
        return np.random.normal(self.mu, self.sigma, nb_samples)

    def eval(self, data, **kwargs):
        metrics = {}
        # metrics['mean'] = np.abs(np.mean(data) - self.mu)
        # metrics['std'] = np.abs(np.std(data, ddof=1) - self.sigma)
        metrics['mean'] = self.mu
        metrics['std'] = self.sigma
        metrics['weisserstein'] = np.linalg.norm(np.mean(data) - self.mu) + np.linalg.norm(np.std(data, ddof=1) - self.sigma)
        return metrics
