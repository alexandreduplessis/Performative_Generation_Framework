import ot
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np


def wasserstein_distance(data1, data2):
    """
    Compute the Wasserstein distance between two datasets

    Parameters
    ----------
    data1 : Numpy array or Torch tensor
        First dataset
    data2 : Numpy array or Torch tensor
        Second dataset

    Returns
    -------
    wasserstein_distance : float
        Wasserstein distance between the two datasets
    """
    if isinstance(data1, torch.Tensor):
        data1 = data1.cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.cpu().numpy()
    M = ot.dist(data1, data2)
    ab = np.ones(data1.shape[0]) / data1.shape[0]
    return ot.emd2(ab, ab, M)

def plot_data(train_data, gen_data):
    train_data = train_data.numpy()
    gen_data = gen_data.numpy()

    plt.scatter(train_data[:, 0], train_data[:, 1], label='train', color='blue')
    plt.scatter(gen_data[:, 0], gen_data[:, 1], label='gen', color='red')

    full_data = np.concatenate([train_data, gen_data], axis=0)
    x_min, x_max = full_data[:, 0].min() * 0.9, full_data[:, 0].max() * 1.1

    y_min, y_max = full_data[:, 1].min() * 0.9, full_data[:, 1].max() * 1.1

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.legend()

    # Display the plot
    plt.show()