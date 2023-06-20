import numpy as np
from sklearn.datasets import make_moons

# TODO recode and integrate two moons in toy_data
def two_moons_dataset(nb_samples=1000, noise=0.1):
    """
    Generate the two moons dataset

    Parameters
    ----------
    nb_samples : int
        Number of samples to generate
    noise : float
        Noise added to the dataset

    Returns
    -------
    data : array
        Generated data
    """
    data = make_moons(n_samples=nb_samples, noise=noise)[0]
    return data
